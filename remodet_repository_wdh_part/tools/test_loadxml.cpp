#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <algorithm>
#include <stdint.h>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
using namespace boost::property_tree;
using namespace std;

int main(int nargc, char** args) {
  const string source = "/home/ethan/DataSets/REID/PRW/Layout/Layout_prw_val.txt";
  std::ifstream infile(source.c_str());
  vector<string> lines;
  std::string str_line;
  while (std::getline(infile, str_line)) {
    lines.push_back(str_line);
  }
  stringstream ss;
  ss.clear();
  ss.str(lines[0]);
  string path;
  int xys[4] = {0};
  int id;
  ss >> path >> id;
  for (int i=0;i<4;i++){
    ss>>xys[i];
  }
  cout<<xys[0]<<" "<<xys[1]<<" "<<xys[2]<<" "<<xys[3];
  return 0;
}