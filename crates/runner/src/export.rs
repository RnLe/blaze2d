//! Native exports use the same descriptors and UTF-8 NPZ manifest as Python.
use std::{fs::File, io::{self, Write}, path::Path};
use blaze2d_interface::{Arrays, DType, ResultRecord, SampleRecord};
use npyz::WriterBuilder;
use serde_json::{Value, json};
use zip::{ZipWriter, write::SimpleFileOptions};
use crate::Study;

type ExportResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn write(study: &Study, path: &Path) -> ExportResult<()> {
    for r in &study.results { r.validate_arrays()?; }
    for e in &study.errors { if let Some(r) = &e.partial_result { r.validate_arrays()?; } }
    match path.extension().and_then(|s|s.to_str()) {
        Some("json"|"ndjson"|"jsonl") => {
            let mut file = io::BufWriter::new(File::create(path)?);
            serde_json::to_writer(&mut file, study)?;
            writeln!(file)?;
            file.flush()?;
        },
        Some("npz") => write_npz(study, path)?,
        _ => return Err("Use a .json, .ndjson, or .npz filename".into()),
    }
    Ok(())
}

fn array_manifest(arrays: &Arrays, zip: &mut ZipWriter<File>, counter: &mut usize) -> ExportResult<Value> {
    let mut output = serde_json::Map::new();
    for (name, a) in arrays {
        let key = format!("array_{:06}", *counter);
        *counter += 1;
        zip.start_file(format!("{key}.npy"), SimpleFileOptions::default())?;
        let shape = a.shape.iter().map(|&n|n as u64).collect::<Vec<_>>();
        match a.dtype {
            DType::Float64 => {
                let mut writer = npyz::WriteOptions::<f64>::new().default_dtype().shape(&shape).writer(&mut *zip).begin_nd()?;
                writer.extend(a.data.iter().copied())?; writer.finish()?;
            },
            DType::Complex128 => {
                let mut writer = npyz::WriteOptions::<num_complex::Complex64>::new().default_dtype().shape(&shape).writer(&mut *zip).begin_nd()?;
                writer.extend(a.data.chunks_exact(2).map(|v|num_complex::Complex64::new(v[0],v[1])))?; writer.finish()?;
            }
        }
        output.insert(name.clone(), json!({"dtype":a.dtype,"shape":a.shape,"dimensions":a.dimensions,
            "order":a.order,"buffer":key}));
    }
    Ok(Value::Object(output))
}

fn sample_manifest(sample: &SampleRecord, zip: &mut ZipWriter<File>, counter: &mut usize) -> ExportResult<Value> {
    Ok(json!({"sample_index":sample.sample_index,"metadata":sample.metadata,
        "arrays":array_manifest(&sample.arrays, zip, counter)?}))
}

fn result_manifest(result: &ResultRecord, zip: &mut ZipWriter<File>, counter: &mut usize) -> ExportResult<Value> {
    let mut value = json!({"schema":result.schema,"task":result.task,"job_index":result.job_index,
        "metadata":result.metadata,"arrays":array_manifest(&result.arrays,zip,counter)?});
    if !result.samples.is_empty() {
        value["samples"] = Value::Array(result.samples.iter().map(|s|sample_manifest(s,zip,counter)).collect::<ExportResult<_>>()?);
    }
    Ok(value)
}

fn write_npz(study: &Study, path: &Path) -> ExportResult<()> {
    let mut zip = ZipWriter::new(File::create(path)?);
    let mut counter = 0;
    let results = study.results.iter().map(|r|result_manifest(r,&mut zip,&mut counter)).collect::<ExportResult<Vec<_>>>()?;
    let mut errors = Vec::new();
    for error in &study.errors {
        let mut value = json!({"job_index":error.job_index,"diagnostic":error.diagnostic});
        if let Some(r) = &error.partial_result { value["partial_result"] = result_manifest(r,&mut zip,&mut counter)?; }
        errors.push(value);
    }
    let manifest = serde_json::to_vec(&json!({"schema":study.schema,"config":study.config,
        "results":results,"errors":errors,"statistics":study.statistics}))?;
    zip.start_file("manifest.npy", SimpleFileOptions::default())?;
    let mut writer = npyz::WriteOptions::<u8>::new().default_dtype().shape(&[manifest.len() as u64]).writer(&mut zip).begin_nd()?;
    writer.extend(manifest)?; writer.finish()?;
    zip.finish()?.sync_all()?;
    Ok(())
}
