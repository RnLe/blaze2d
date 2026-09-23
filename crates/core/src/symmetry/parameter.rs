//! Deterministic parameter-job ordering with explicit coordinate scales.
//! Reordering changes continuation, never the physical job set or output order.
use serde::Serialize;

pub mod sweep;

#[derive(Debug, Clone, Serialize)]
pub struct ParameterMetric {
    scales: Vec<f64>,
}
impl ParameterMetric {
    pub fn new(scales: Vec<f64>) -> Result<Self, String> {
        if scales.is_empty() || scales.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("parameter scales must be finite and positive".into());
        }
        Ok(Self { scales })
    }
    pub fn validate(&self, coordinates: &[f64]) -> Result<(), String> {
        if coordinates.len() != self.scales.len() || coordinates.iter().any(|x| !x.is_finite()) {
            return Err("parameter coordinates do not match the finite scaled metric".into());
        }
        Ok(())
    }
    pub fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, String> {
        self.validate(a)?;
        self.validate(b)?;
        let distance = a
            .iter()
            .zip(b)
            .zip(&self.scales)
            .fold(0.0_f64, |r, ((a, b), scale)| {
                let difference = a - b;
                let scaled = if difference.is_finite() {
                    difference / scale
                } else {
                    a / scale - b / scale
                };
                r.hypot(scaled)
            });
        if !distance.is_finite() {
            return Err("scaled parameter distance overflow".into());
        }
        Ok(distance)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterOrder {
    Given,
    Nearest,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParameterRoute {
    pub policy: ParameterOrder,
    /// Indices into the original requested jobs; duplicates remain distinct.
    pub order: Vec<usize>,
    pub scaled_distance: f64,
    pub original_scaled_distance: f64,
}
impl ParameterRoute {
    /// The first job is fixed. Nearest routing is an O(n²), 4096-job reference.
    /// Distances are diagnostic predictors, not an asserted solver cost model.
    pub fn new(
        parameters: &[Vec<f64>],
        metric: &ParameterMetric,
        policy: ParameterOrder,
    ) -> Result<Self, String> {
        if parameters.len() > 4096 {
            return Err("parameter routing is limited to 4096 jobs".into());
        }
        for p in parameters {
            metric.validate(p)?;
        }
        let route_length = |order: &[usize]| -> Result<f64, String> {
            let sum = order.windows(2).try_fold(0.0, |sum, pair| {
                Ok::<_, String>(sum + metric.distance(&parameters[pair[0]], &parameters[pair[1]])?)
            })?;
            if !sum.is_finite() {
                return Err("parameter route length overflow".into());
            }
            Ok(sum)
        };
        let original: Vec<_> = (0..parameters.len()).collect();
        let original_scaled_distance = route_length(&original)?;
        let mut order = Vec::with_capacity(parameters.len());
        if policy == ParameterOrder::Given || parameters.is_empty() {
            order = original;
        } else {
            let mut visited = vec![false; parameters.len()];
            let mut current = 0;
            while order.len() < parameters.len() {
                order.push(current);
                visited[current] = true;
                let mut next: Option<(f64, usize)> = None;
                for i in 0..parameters.len() {
                    if visited[i] {
                        continue;
                    }
                    let distance = metric.distance(&parameters[current], &parameters[i])?;
                    if next.is_none_or(|(d, j)| distance.total_cmp(&d).then(i.cmp(&j)).is_lt()) {
                        next = Some((distance, i));
                    }
                }
                if let Some((_, i)) = next {
                    current = i;
                }
            }
        }
        Ok(Self {
            policy,
            scaled_distance: route_length(&order)?,
            original_scaled_distance,
            order,
        })
    }

    /// Restore any complete output type, including physical fields, to the
    /// original job order. A mutated or incomplete permutation is rejected.
    pub fn restore<T>(&self, execution_outputs: Vec<T>) -> Result<Vec<T>, String> {
        if execution_outputs.len() != self.order.len() {
            return Err("parameter output count differs from the requested jobs".into());
        }
        let mut restored: Vec<Option<T>> = std::iter::repeat_with(|| None)
            .take(self.order.len())
            .collect();
        for (&index, output) in self.order.iter().zip(execution_outputs) {
            let slot = restored
                .get_mut(index)
                .ok_or("invalid parameter output index")?;
            if slot.is_some() {
                return Err("duplicate parameter output index".into());
            }
            *slot = Some(output);
        }
        restored
            .into_iter()
            .map(|x| x.ok_or_else(|| "missing parameter output".into()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_route_keeps_duplicate_jobs_and_restores_outputs() {
        let jobs = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 5.0],
        ];
        let route = ParameterRoute::new(
            &jobs,
            &ParameterMetric::new(vec![1.0, 10.0]).unwrap(),
            ParameterOrder::Nearest,
        )
        .unwrap();
        assert_eq!(route.order, [0, 2, 3, 1]);
        assert_eq!(
            route.restore(vec!["a", "a-repeat", "c", "b"]).unwrap(),
            ["a", "b", "a-repeat", "c"]
        );
        assert!(route.restore(vec![0]).is_err());
        assert!(route.scaled_distance < route.original_scaled_distance);
        assert!(ParameterMetric::new(vec![0.0]).is_err());
        assert!(
            ParameterRoute::new(
                &[vec![f64::NAN]],
                &ParameterMetric::new(vec![1.0]).unwrap(),
                ParameterOrder::Given
            )
            .is_err()
        );
        assert!(
            ParameterRoute::new(
                &[vec![f64::MAX], vec![-f64::MAX]],
                &ParameterMetric::new(vec![1.0]).unwrap(),
                ParameterOrder::Nearest
            )
            .is_err()
        );
    }
}
