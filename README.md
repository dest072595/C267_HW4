# C267_HW4
Spring 2024 Computer Science C267 Homework #4



Google doc for write up: https://docs.google.com/document/d/14R7rZA-ijM14Io373i5g7BQV4bQpZDz5eeZq65qHbXQ/edit?usp=sharing



Recitation video: 
https://drive.google.com/file/d/1uv1LPE1xFBrQEWYGo-imxd1puaKZDHfL/view


slides: 
https://docs.google.com/document/d/1sKpsYHH50nTSLpUcN92rawD1fAKyrt8PtYjPfgWFhXk/edit?usp=sharing

Chapter 16 from recitation: 
http://www.math.iit.edu/~fass/477577_Chapter_16.pdf


sparse matrix wiki:
https://en.wikipedia.org/wiki/Sparse_matrix


jacobi preconditioner:
https://phtournier.pages.math.cnrs.fr/5mm29/blockjacobi/


lecture 19: 
https://drive.google.com/file/d/1Osre3sqZopUTYPgAequf4gDY66AlueH3/view


Running the code: 


- Destinee had to download Eigen as a tar.gz file and put it into an Include directory in the repo 
- Eigen and MPI libraries should be preinstalled if not install them and change paths on cmakelist.txt for MPI below

- find path to MPI and replace the path below(below is for MAC on Destinee's computer')
- add to cmake file under "find_package(MPI REQUIRED)"

find_package(MPI REQUIRED)

if(MPI_FOUND)
    include_directories(/opt/homebrew/Cellar/open-mpi/5.0.2_1/include)
    link_directories(/opt/homebrew/Cellar/open-mpi/5.0.2_1/lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/eigen3)

- I couldnt get this command to run:  cmake -DCMAKE_BUILD_TYPE=Release .. 
- if your Eigen and MPI libraries are installed correctly run the command above for cmake in a new directory "build" then can run make as we did for previous assignments. 

- so I manually compiled using the command below, might have to change include or remove it based on your computer and where the files are. Mine was in the Include directory and I compiled with this: 

mpic++ -o output distributed_pcg.cpp -Wall -std=c++17 -I./Include/eigen-3.3.7
mpirun -np 1 ./output    

- change number of processes- may be different if ran in Release mode instead?

mpirun -np 2 ./output    
mpirun -np 4 ./output    

mpirun -np 8 ./output 

change the size of matrix in code "N = some #"


- *** Results of serial code pasted in google doc



