from mpi4py import MPI
from server import FederatedServer
from client import FederatedClient

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Server process
        FederatedServer(comm).run()
    else:
        # Client process
        FederatedClient(comm, rank).run()

if __name__ == "__main__":
    main()
