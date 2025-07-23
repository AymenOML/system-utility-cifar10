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

if __name__ == '__main__':
    main()
    # Finalization logic
    MPI.Finalize()
