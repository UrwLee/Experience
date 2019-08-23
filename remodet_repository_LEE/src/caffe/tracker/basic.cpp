#include "caffe/tracker/basic.hpp"

#include <string>
#include <cstdio>
#include <vector>

namespace caffe {

namespace bfs = boost::filesystem;

using std::string;
using std::sprintf;
using std::vector;

// Maximum length of a string created from a number.
const int kMaxNum2StringSize = 50;

// Default number of decimal places when converting a double to a string.
const int kDefaultDecimalPlaces = 6;

string int2str(const int num) {
 char num_buffer[kMaxNum2StringSize];
 sprintf(num_buffer, "%d", num);
 return num_buffer;
}

string num2str(const int num) {
  return int2str(num);
}

string double2str(const double num, const int decimal_places) {
  const string format = "%." + num2str(decimal_places) + "lf";
  char num_buffer[kMaxNum2StringSize];
  sprintf(num_buffer, format.c_str(), num);
  return num_buffer;
}

string float2str(const float num) {
 char num_buffer[kMaxNum2StringSize];
 sprintf(num_buffer, "%f", num);
 return num_buffer;
}

string unsignedint2str(const unsigned int num) {
  char num_buffer[kMaxNum2StringSize];
  sprintf(num_buffer, "%u", num);
  return num_buffer;
}

string num2str(const double num) {
  return double2str(num, kDefaultDecimalPlaces);
}

string num2str(const double num, int decimal_places) {
  return double2str(num, decimal_places);
}

string num2str(const float num) {
  return float2str(num);
}

string num2str(const unsigned int num) {
  return unsignedint2str(num);
}

string num2str(const size_t num) {
  char num_buffer[50];
  sprintf(num_buffer, "%zu", num);
  return num_buffer;
}

// *******File IO *************
void find_subfolders(const bfs::path& folder, vector<string>* sub_folders) {
  if (!bfs::is_directory(folder)) {
    LOG(FATAL) << "Error - " << folder.c_str() << " is not a valid directory!";
    return;
  }

  bfs::directory_iterator end_itr; // default construction yields past-the-end
  for (bfs::directory_iterator itr(folder); itr != end_itr; ++itr) {
    if (bfs::is_directory(itr->status())) {
      string filename = itr->path().filename().string();
      sub_folders->push_back(filename);
    }
  }
  // Sort the files by name.
  std::sort(sub_folders->begin(), sub_folders->end());
}

void find_matching_files(const bfs::path& folder, const boost::regex filter,
                         vector<string>* files) {
  if (!bfs::is_directory(folder)) {
    LOG(FATAL) << "Error - " << folder.c_str() << " is not a valid directory!";
    return;
  }

  bfs::directory_iterator end_itr; // default construction yields past-the-end
  for (bfs::directory_iterator itr(folder); itr != end_itr; ++itr) {
    if (bfs::is_regular_file(itr->status())) {
      string filename = itr->path().filename().string();

      boost::smatch what;
      if(boost::regex_match(filename, what, filter) ) {
        files->push_back(filename);
      }
    }
  }
  // Sort the files by name.
  std::sort(files->begin(), files->end());
}

float sample_rand_uniform() {
  // Generate a random number in (0,1)
  return (float)((rand() + 1) / (static_cast<float>(RAND_MAX) + 2));
}

template<typename Dtype>
Dtype sample_exp(const Dtype lambda) {
  const Dtype rand_uniform = sample_rand_uniform();
  return -log(rand_uniform) / lambda;
}
template float sample_exp(const float lambda);
template double sample_exp(const double lambda);

template<typename Dtype>
Dtype sample_exp_two_sided(const Dtype lambda) {
  const Dtype pos_or_neg = (rand() % 2 == 0) ? 1 : -1;
  const Dtype rand_uniform = sample_rand_uniform();
  return log(rand_uniform) / lambda * pos_or_neg;
}
template float sample_exp_two_sided(const float lambda);
template double sample_exp_two_sided(const double lambda);

}
