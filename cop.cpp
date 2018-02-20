#include <vector> 
#include <fstream>
#include <random>
#include <assert.h>
#include <stdio.h>
#include <math.h> 
#define PI 3.14159265

// Rotate the configurations by 90 degrees (x->y,y->x)
int rotate(const unsigned int i, const unsigned int L){
  return (i%L)*L + i/L;
}

// Fetch a list of nearest neighbors
std::vector< int > getNeighbors(const unsigned int site, const unsigned int L){
  int x=site/L;
  int y=site%L;
  std::vector< int > sites(4);
  sites[0] = ((y+1+L)%L) +           x*L;
  sites[1] = ((y-1+L)%L) +           x*L;
  sites[2] =           y + ((x+1+L)%L)*L;
  sites[3] =           y + ((x-1+L)%L)*L;
  return sites;
}

// Get the x,y coordinates of a lattice site
std::vector< int > getXY(const unsigned int site, const unsigned int L){
  std::vector< int > xy(2);
  xy[0]=site/L;
  xy[1]=site%L;
  return xy;
}

//main function
int main(int argc, char** argv){
  
  //Read config.txt to get the parameters
  unsigned int sweeps=100000;
  unsigned int thermalization=100000;
  unsigned int nskip=100000;
  unsigned int L=10;
  unsigned int seed=time(0);
  double T=1.6;
  unsigned int writeConfigs=0;
  unsigned int calculateOP=0;
  if(argc>1){
    seed=atoi(argv[1]);
  }
  if(argc>2){
    T=atof(argv[2]);
  }
  if(argc>3){
    L=atoi(argv[3]);
  }
  
  
  std::ifstream cFile ("config.txt");
  if (cFile.is_open())
  {
    std::string line;
    while(getline(cFile, line)){
      if(line[0] == '#' || line.empty())
          continue;
      int delimiterPos = line.find("=");
      std::string name = line.substr(0, delimiterPos);
      std::string value = line.substr(delimiterPos + 1);
      if(name == "sweeps"){
        sweeps = std::atoi(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "thermalization"){
        thermalization = std::atoi(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "nskip"){
        nskip = std::atoi(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "L"){
        L = std::atoi(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "seed"){
        seed = std::atoi(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "T"){
        T = std::atof(line.substr(delimiterPos + 1).c_str());
      }
      else if(name == "configs"){
        writeConfigs=1;
      }
      else if(name == "op"){
        calculateOP=1;
      }
    }
    cFile.close();
  }
  
  // Initialize the random number generator
  std::mt19937 generator (seed);
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  
  // Fill an initial spin confguration in the Sz=0 sector
  std::vector< int > spins(L*L,-1);
  do{
    int site = int(L*L*dis(generator));
    spins[site]=1;
  } while (std::accumulate(spins.begin(),spins.end(), 0)< 0);
  
  // Prepare the list of ups and downs
  std::vector< int > up;
  std::vector< int > dn;
  for (int i=0;i<L*L;i++){
    if( spins[i] == 1){ up.push_back(i);}
    else {dn.push_back(i);}
  }
  
  
  // Do the MC sweeps
  std::vector< int > nbrs;
  int length=0;
  if (writeConfigs){length=sweeps/nskip;}
  std::vector<std::vector< int > > images(length,std::vector<int>(L*L));
  if (!calculateOP){length=0;}else{length=sweeps/nskip;}
  std::vector< double > op(length);
  int counter=0;
  for (int mcs=0;mcs<(sweeps+thermalization)*L*L;mcs++){
    
    // With 5% probability, rotate the configurations
    if (dis(generator)<0.05){
      for (int i=0;i<L*L;++i){
        int j = rotate(i,L);
        if (i<j && spins[i]!=spins[j]){
          std::swap(spins[i],spins[j]);
        }
      }
      
      up.clear();
      dn.clear();
      for (int i=0;i<L*L;i++){
        if( spins[i] == 1){ up.push_back(i);}
        else {dn.push_back(i);}
      }
    } 
    // Sz preserving MC update
    else{
      
      // Choose two sites
      int i = int(dis(generator)*up.size());
      int j = int(dis(generator)*dn.size());
      int si = up[i];
      int sj = dn[j];
      assert(si!=sj);
      assert(spins[si]!=spins[sj]);
      
      // Calculate the energy difference of the proposed state
      double de = 0.0;
      nbrs=getNeighbors(si,L);
      for(std::vector< int >::iterator it = nbrs.begin(); it != nbrs.end(); ++it) {
        if( *it != sj ){de += 2*spins[*it]*spins[si];}
      }
      nbrs=getNeighbors(sj,L);
      for(std::vector< int >::iterator it = nbrs.begin(); it != nbrs.end(); ++it) {
        if( *it != si ){de += 2*spins[*it]*spins[sj];}
      }
      // Decide if we accept the update
      if(de<0.0 || std::exp(-de/T) > dis(generator)){
        std::swap(spins[si],spins[sj]);
        std::swap(up[i],dn[j]);
      }
    }
    
    // Perform the measurements and write the configuration
    if ((mcs >= thermalization*L*L) && (mcs)%(L*L*nskip)==0){
      if(calculateOP){
        double s=0;
        for(int i=0 ; i < L*L; i++){
          for(int j=0; j < L*L; j++){
            std::vector< int > xya=getXY(i,L);
            std::vector< int > xyb=getXY(j,L);
            s+=spins[i]*spins[j]*(cos(2*PI/L*(xya[0]-xyb[0]))+cos(2*PI/L*(xya[1]-xyb[1])));
          }
        }
        op[counter]=s/L/L/L/L;
      }
      if(writeConfigs>0){
        images[counter]=spins;
      }
      counter++;
    }
  }
  //Prepare the output files
  char buffer [50];
  sprintf(buffer,"L=%d-T=%.02f-%d.txt",L,T,seed);
  std::string name=buffer;
  std::ofstream output(name);
  if(output.is_open()){
    output << "# L=" << L <<", T=" << T << ", seed=" << seed;
    output <<  ", nskip=" << nskip << ", thermalization=" << thermalization;
    output <<  ", sweeps=" << sweeps << std::endl;
  
    for(int i=0;i<counter;i++){
      if(writeConfigs){
        for (int j=0;j<L*L;j++){
          output << images[i][j] << " ";
        }
      }
      if(calculateOP){
        output << op[i] << " ";
      }
      output << T << std::endl;
    }
  }
  output.close();
  exit(0);
}
  
  
  
