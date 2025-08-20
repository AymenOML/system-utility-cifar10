from mpi4py import MPI
from mpi_server import run_server
from mpi_client import run_client
from postprocess_pipeline import run_postprocessing

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        run_server(comm)
    else:
        run_client(comm, rank)

    # make sure everyone is done before postprocessing
    comm.Barrier()

    # run normalization + labeling once on rank 0
    if rank == 0:
        try:
            out = run_postprocessing(base_dir="Data")  # <-- correct kw name
            print(f"[Postprocess] Saved: {out}")
        except Exception as e:
            print(f"[Postprocess] Skipped: {e}")

    # clean exit for all ranks
    comm.Barrier()

if __name__ == "__main__":
    main()
