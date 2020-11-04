from functools import partial
import multiprocessing
import time

oi = 1


manager = multiprocessing.Manager()
ship_ = manager.list()  # Shared Proxy to a list    
def method(args):
    num = args[0] + 1
    indice = args[1]
    ship_.append([num, indice])
    time.sleep(1)


if __name__ == "__main__":

    argumentBag = [[1,0], [3,1], [5,2], [7,3]]

    ship = [[1,1], [1,1], [1,1], [1,1]]

    for i in range(5):

        with multiprocessing.Pool(processes=3) as pool:
            pool.map(method, argumentBag)
            pool.close()
            pool.join()    
        
        print("Ship global")
        print(ship_)     

        for i in range(4):
            argumentBag[ship_[i][1]][0] = ship_[i][0]
        ship_[:] = []

        print("Ship local")
        print(argumentBag)