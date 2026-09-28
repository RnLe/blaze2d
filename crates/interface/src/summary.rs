//! Compact reporting derived from the retained numerical execution ledger.
use crate::{ResultRecord, Task};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn count(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.to_owned()).or_default() += 1;
}

fn entries(value: &Value) -> impl Iterator<Item = &Value> {
    value.as_array().into_iter().flatten()
}

/// Summarize completed outputs and all attempted work, including a failed point.
/// Coordinate counts refer to solved blocks; reuse does not invent a new solve.
pub fn record(result: &mut ResultRecord) {
    if result.task != Task::Bands {
        return;
    }
    let m = &result.metadata;
    let requested = &m["config"]["bands"]["symmetry"];
    let policy = requested["policy"].as_str().unwrap_or("manual");
    let mut methods = BTreeMap::new();
    let mut coordinates = BTreeMap::new();
    let mut reasons = BTreeMap::new();
    let (
        mut reused,
        mut geometry_seeds,
        mut coarse_seeds,
        mut rejected_attempts,
        mut attempted_iterations,
        mut retained_iterations,
        mut fallback_points,
    ) = (0, 0, 0, 0, 0, 0, 0);
    let returned = result
        .arrays
        .get("frequencies")
        .and_then(|a| a.shape.first())
        .copied()
        .unwrap_or(0);
    let failed_index = m["failed_point"]["k_index"].as_u64().map(|n| n as usize);
    let completed = failed_index.map_or(returned, |n| n.min(returned));
    let failed = m
        .get("failed_point")
        .and_then(|p| p.get("symmetry"))
        .filter(|p| !p.is_null());
    let mut reports: Vec<_> = entries(&m["symmetry"]["points"]).collect();
    // A failed metric point may already have emitted its partial field record.
    // Its final failure ledger replaces that entry, rather than adding its work twice.
    if let Some(failed) = failed {
        if let Some(index) = failed_index.filter(|&index| index < reports.len()) {
            reports[index] = failed;
        } else {
            reports.push(failed);
        }
    }
    for (index, p) in reports.into_iter().enumerate() {
        let mut point_reasons = BTreeSet::new();
        for reason in entries(&p["rejected"]).filter_map(Value::as_str) {
            point_reasons.insert(reason);
        }
        for reason in [
            p["continuation"]["fallback"].as_str(),
            p["coarse_start"]["rejection"].as_str(),
        ]
        .into_iter()
        .flatten()
        {
            point_reasons.insert(reason);
        }
        attempted_iterations += p["attempted_iterations"].as_u64().unwrap_or(0);
        retained_iterations += p["retained_iterations"].as_u64().unwrap_or(0);
        geometry_seeds += usize::from(!p["continuation"]["seed"].is_null());
        coarse_seeds += usize::from(p["coarse_start"]["used_as_seed"] == true);
        for attempt in entries(&p["solver_attempts"]) {
            if let Some(reason) = attempt["rejection"].as_str() {
                rejected_attempts += 1;
                point_reasons.insert(reason);
            }
            if index < completed && attempt["retained"] == true {
                count(
                    &mut methods,
                    attempt["method"].as_str().unwrap_or("unknown"),
                );
            }
        }
        if index < completed {
            if !p["source_index"].is_null() || !p["continuation"]["exact_source"].is_null() {
                reused += 1;
            }
            if let Some(name) = p["coordinates"]["representation"].as_str() {
                count(&mut coordinates, name);
            }
            for sector in entries(&p["sectors"]["sectors"]) {
                count(
                    &mut coordinates,
                    sector["coordinates"]["representation"]
                        .as_str()
                        .unwrap_or("sector_specific"),
                );
                if let Some(reason) = sector["coordinate_rejection"].as_str() {
                    point_reasons.insert(reason);
                }
            }
        }
        // Includes declined reuse/cache admission and suppressed repeated PT
        // attempts. A successful coarse initializer alone is not a fallback.
        if !point_reasons.is_empty() {
            fallback_points += 1;
        }
        for reason in point_reasons {
            count(&mut reasons, reason);
        }
    }
    if m.get("external_dielectric").is_some() {
        count(&mut methods, "sampled_spatial_complex");
        attempted_iterations = entries(&m["iterations"]).filter_map(Value::as_u64).sum();
        retained_iterations = attempted_iterations;
    }
    let accuracy = m["accuracy"]["status"].as_str().unwrap_or("unavailable");
    let converged = m["converged"] == true;
    let describe = |counts: &BTreeMap<String, usize>| {
        counts
            .iter()
            .map(|(s, n)| format!("{s}={n}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let requested_text = if policy == "auto" {
        "auto".to_owned()
    } else {
        format!(
            "manual (arithmetic={}, reuse={}, sectors={}, continuation={})",
            requested["arithmetic"]
                .as_str()
                .unwrap_or("spatial_complex"),
            requested["k_reuse"].as_str().unwrap_or("off"),
            requested["sectors"].as_str().unwrap_or("off"),
            requested["continuation"].as_str().unwrap_or("off")
        )
    };
    let mut text = format!(
        "Symmetry {requested_text}: retained [{}]{}; {reused}/{completed} points reused; {fallback_points} points with fallback or declined optimization. Stopping {}; accuracy {accuracy}.",
        describe(&methods),
        if coordinates.is_empty() {
            String::new()
        } else {
            format!("; coordinates [{}]", describe(&coordinates))
        },
        if converged { "converged" } else { "incomplete" }
    );
    if let Some((reason, n)) = reasons.iter().next() {
        text.push_str(&format!(
            " Reason ({n} points): {reason}{}.",
            if reasons.len() > 1 {
                "; further reasons in metadata"
            } else {
                ""
            }
        ));
    } else if policy == "auto" && coordinates.is_empty() {
        if let Some(reason) = entries(&m["symmetry"]["automatic"]["reasons"])
            .next()
            .and_then(Value::as_str)
        {
            text.push_str(&format!(" {reason}."));
        }
    }
    result.metadata["execution"] = json!({
        "version":1, "text":text, "requested":requested,
        "resolved":m["symmetry"].get("resolved").unwrap_or(requested),
        "completed_points":completed, "returned_points":returned, "stopping_converged":converged, "accuracy":accuracy,
        "retained_solve_methods":methods, "coordinate_blocks":coordinates,
        "reused_points":reused, "geometry_seeded_points":geometry_seeds, "coarse_seeded_points":coarse_seeds,
        "attempted_iterations":attempted_iterations, "retained_iterations":retained_iterations,
        "rejected_solver_attempts":rejected_attempts, "fallback_or_declined_points":fallback_points,
        "fallback_reasons":reasons, "policy_reasons":m["symmetry"]["automatic"]["reasons"],
        "details":"symmetry.points, discretization.points and failed_point include qualification, preparation and rejected work"
    });
}

/// Bound terminal output for large studies while retaining every JSON report.
pub fn format_results<'a>(results: impl IntoIterator<Item = &'a ResultRecord>) -> String {
    let mut groups = BTreeMap::new();
    for r in results {
        if let Some(text) = r.metadata["execution"]["text"].as_str() {
            count(&mut groups, text);
        }
    }
    let mut lines: Vec<_> = groups
        .iter()
        .take(8)
        .map(|(text, n)| format!("{n} job(s): {text}"))
        .collect();
    if groups.len() > 8 {
        lines.push(format!(
            "{} further execution summaries in result metadata.",
            groups.len() - 8
        ));
    }
    lines.join("\n")
}
