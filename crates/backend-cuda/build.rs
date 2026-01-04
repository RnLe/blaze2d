//! Build script for blaze2d-backend-cuda.
//!
//! This script locates the CUDA toolkit and configures linking against libcufft.

fn main() {
    // Only do CUDA linking when the cuda feature is enabled
    #[cfg(feature = "cuda")]
    {
        link_cufft();
    }
}

#[cfg(feature = "cuda")]
fn link_cufft() {
    // Try to find CUDA installation path in order of preference:
    // 1. CUDA_PATH environment variable (Windows convention)
    // 2. CUDA_HOME environment variable (Linux convention)
    // 3. Default paths
    let cuda_path = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            // Check common default locations
            let default_paths = [
                "/usr/local/cuda",
                "/opt/cuda",
                "/usr/lib/cuda",
            ];
            
            for path in default_paths {
                if std::path::Path::new(path).exists() {
                    return path.to_string();
                }
            }
            
            // Fall back to default, let the linker find it
            "/usr/local/cuda".to_string()
        });

    // Determine library path based on architecture
    let lib_path = if cfg!(target_os = "windows") {
        format!("{}/lib/x64", cuda_path)
    } else if cfg!(target_arch = "x86_64") {
        // Try lib64 first (common on Linux), fall back to lib
        let lib64 = format!("{}/lib64", cuda_path);
        if std::path::Path::new(&lib64).exists() {
            lib64
        } else {
            format!("{}/lib", cuda_path)
        }
    } else {
        format!("{}/lib", cuda_path)
    };

    // Tell cargo to look for libraries in CUDA lib directory
    println!("cargo:rustc-link-search=native={}", lib_path);
    
    // Link against cuFFT library
    // On Linux this links libcufft.so, on Windows cufft.lib
    println!("cargo:rustc-link-lib=cufft");

    // Rerun build script if these environment variables change
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
