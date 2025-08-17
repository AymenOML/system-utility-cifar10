from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        from mpi_server import run_server
        run_server(comm)
    else:
        from mpi_client import run_client
        run_client(comm, rank)

    # Ensure all data/plots/CSVs are written by every rank
    comm.Barrier()

    # Run post-processing only on the server
    if rank == 0:
        from federated_mpi.postprocess_pipeline import run_postprocessing
        # Set fail_job_on_error=True if you prefer the job to FAIL when post steps fail
        run_postprocessing(base_data_dir="Data", fail_job_on_error=False)

    # Optional: a second barrier to keep everyone in lockstep
    comm.Barrier()

if __name__ == '__main__':
    main()
    MPI.Finalize()
