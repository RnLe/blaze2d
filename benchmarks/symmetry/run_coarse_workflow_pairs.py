"""Run the existing paired sampler with the stricter coarse-only contract."""
import run_pairs
from coarse_workflow_contract import configuration, path_issues, same_jobs


def main():
    quality = run_pairs.quality_passes
    original_configuration = run_pairs.physical_configuration
    original_jobs = run_pairs.same_interface_jobs
    try:
        run_pairs.physical_configuration = configuration
        run_pairs.same_interface_jobs = same_jobs
        run_pairs.quality_passes = lambda result, bands, field_output=True: (
            quality(result, bands, field_output) and not path_issues(result))
        run_pairs.main()
    finally:
        run_pairs.physical_configuration = original_configuration
        run_pairs.same_interface_jobs = original_jobs
        run_pairs.quality_passes = quality


if __name__ == '__main__':
    main()
