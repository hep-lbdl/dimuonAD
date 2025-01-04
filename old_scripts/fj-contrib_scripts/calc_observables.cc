// Adapted from example_basic_usage
// run: ./calc_observables < /path/to/event/file.dat

// This must be run in the folder /fjcontrib-1.052/Nsubjettiness

#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include <sstream>
#include "Nsubjettiness.hh" // In external code, this should be fastjet/contrib/Nsubjettiness.hh
#include "Njettiness.hh"
#include "NjettinessPlugin.hh"
#include "SoftDrop.hh"

using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

// sim
//string input_file_path = "/global/u1/r/rmastand/dimuonAD/data_pre_fj/hadrons_only_sim_zmm_forcms_1k-mz90.1-mw80.4_8000030.dat";
//string output_file_path = "/global/u1/r/rmastand/dimuonAD/data_post_fj/jet_obs_sim_zmm_forcms_1k-mz90.1-mw80.4_8000030.dat";

// forward declaration to make things clearer
vector<vector<PseudoJet>> read_event(string infile_path);
void analyze(const vector<PseudoJet> & input_particles, ofstream& outfile);

//----------------------------------------------------------------------
int main(int argc, char *argv[]){
    
    string start = argv[1];
    string stop = argv[2];
    
    // data
    string input_file_path = "/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_pre_fj/hadrons_only_"+start+"_"+stop+"_od.dat";
    string output_file_path = "/global/cfs/cdirs/m3246/rmastand/dimuonAD/data_post_fj/jet_obs_"+start+"_"+stop+"_od.dat";

    cout << "Reading in input file " << input_file_path << "..." << endl;
    cout << "Outputting to " << output_file_path << "." << endl;

    // Read in events
    vector<vector<PseudoJet>> all_events = read_event(input_file_path);

    // Test correct output
    int num_events = all_events.size();
    
    /*
    for (int e = 0; e < num_events; e++){
        cout << "On event " << e << endl;
        int num_particles = all_events[e].size();
        for (int p = 0; p < num_particles; p++){
            cout << "On particle " << p << endl;
            cout <<  "px " << all_events[e][p].px() <<  " py " << all_events[e][p].py() << " pz " << all_events[e][p].pz() << " E "<< all_events[e][p].E() << endl;
        }
    }
    */
    
    // Define an output file
    // output order: tau1, tau2, tau3, tau21, tau32
    ofstream outfile;
    outfile.open(output_file_path);      

    // Calculate nsubjettiness
    for (int e = 0; e < num_events; e++){
        outfile << "#BEGIN" << endl;
        analyze(all_events[e], outfile);
        outfile << "#END" << endl;
    }
    
    outfile.close();
    
        
    cout << "Done!" << endl;
    
  return 0;
    
}

// read in input particles
vector<vector<PseudoJet>> read_event(string infile_path){  
    
    // Initialize a vector to hold the events
    // Each vector is a vector of PseudoJets (particles)
    vector<vector<PseudoJet>> all_events;
    fstream data_file;
    int num_events_read = 0;
    vector<PseudoJet> current_event;
   
    data_file.open(infile_path, ios::in);
    if (data_file.is_open()) {
        string line;
        vector<PseudoJet> current_event;
        
        while (getline(data_file, line)) {
            istringstream linestream(line);
            // take substrings to avoid problems when there are extra "pollution"
            // characters (e.g. line-feed).
            if (line.substr(0,6) == "#BEGIN") {
                // Marks a new event
                vector<PseudoJet> current_event;
            }
            else if  (line.substr(0,4) == "#END") {
                // Marks the end of the event
                all_events.push_back(current_event);  
                current_event.clear();
                num_events_read += 1;
            }
            else {
                double px,py,pz,E;
                linestream >> px >> py >> pz >> E;
                PseudoJet current_particle(px,py,pz,E);
                //push event onto back of full_event vector
                current_event.push_back(current_particle);
            }
        } 
        data_file.close();
    }
    
    cout << "Read in " << num_events_read << " events." << endl;
    return all_events;
}
  


////////
//
//  Main Routine for Analysis 
//
///////

void analyze(const vector<PseudoJet> & input_particles, ofstream& outfile) {
   
   ////////
   //
   //  Start of analysis.  First find anti-kT jets, then find N-subjettiness values of those jets
   //
   ///////
   
   // Initial clustering with anti-kt algorithm
   JetAlgorithm algorithm = antikt_algorithm;
   double jet_rad = 1.00; // jet radius for anti-kt algorithm
   JetDefinition jetDef = JetDefinition(algorithm, jet_rad, E_scheme, Best);
   ClusterSequence clust_seq(input_particles, jetDef);
   vector<PseudoJet> antikt_jets  = sorted_by_pt(clust_seq.inclusive_jets());
   
   for (int j = 0; j < 1; j++) { // Only look at the leading jet

      // get the jet for analysis
      PseudoJet this_jet = antikt_jets[j];
       
       // run SoftDrop
      double z_cut = 0.10;
      double beta_sd  = 0.0;
      contrib::SoftDrop sd(beta_sd, z_cut);
       
      PseudoJet sd_jet = sd(this_jet);
    
      // run Nsub
      double beta_nsub = 1.0;
      NsubjettinessRatio nSub21_beta1(2,1, OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta_nsub));
      
      // calculate Nsubjettiness values (beta = 1.0)
      double tau21_beta1 = nSub21_beta1(sd_jet);
       
       // now loop through all options
      outfile << setprecision(16) << right << fixed;

      // Output: pt eta phi M tau21
      outfile << setw(30) << sd_jet.perp()
         << setw(30) << sd_jet.eta()
         << setw(30) << sd_jet.phi_std()
         << setw(30) << sd_jet.m()
         << setw(30) << tau21_beta1
         << endl;

   }

  
}
