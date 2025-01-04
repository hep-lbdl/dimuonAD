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
#include "XConePlugin.hh"

using namespace std;
using namespace fastjet;
using namespace fastjet::contrib;

string events_file_path = "/global/u1/r/rmastand/dimuonAD/test_event.dat";
string output_file_path = "/global/u1/r/rmastand/dimuonAD/test_output.txt";

// forward declaration to make things clearer
vector<vector<PseudoJet>> read_event(string infile_path);
void analyze(const vector<PseudoJet> & input_particles, ofstream& outfile);

//----------------------------------------------------------------------
int main(){

    // Read in events
    vector<vector<PseudoJet>> all_events = read_event(events_file_path);

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
   JetDefinition jetDef = JetDefinition(algorithm,jet_rad,E_scheme,Best);
   ClusterSequence clust_seq(input_particles,jetDef);
   vector<PseudoJet> antikt_jets  = sorted_by_pt(clust_seq.inclusive_jets());
   
   for (int j = 0; j < 2; j++) { // Two hardest jets per event

      // get the jet for analysis
      PseudoJet this_jet = antikt_jets[j];
      
      cout << "-------------------------------------------------------------------------------------" << endl;
      cout << "Analyzing Jet " << j + 1 << ":" << endl;
      cout << "-------------------------------------------------------------------------------------" << endl;
      
      ////////
      //
      //  Basic checks of tau values first
      //
      //  If you don't want to know the directions of the subjets,
      //  then you can use the simple function Nsubjettiness.
      //
      //  Recommended usage for Nsubjettiness:
      //  AxesMode:  KT_Axes(), WTA_KT_Axes(), OnePass_KT_Axes(), or OnePass_WTA_KT_Axes()
      //  MeasureMode:  Unnormalized_Measure(beta)
      //  beta with KT_Axes: 2.0
      //  beta with WTA_KT_Axes: anything greater than 0.0 (particularly good for 1.0)
      //  beta with OnePass_KT_Axes or OnePass_WTA_KT_Axes:  between 1.0 and 3.0
      //
      ///////
      
      
      cout << "-------------------------------------------------------------------------------------" << endl;
      cout << "N-subjettiness with Unnormalized Measure (in GeV)" << endl;
      cout << "beta = 1.0:  One-pass Winner-Take-All kT Axes" << endl;
      cout << "beta = 2.0:  One-pass E-Scheme kT Axes" << endl;
      cout << "-------------------------------------------------------------------------------------" << endl;
      
      // Now loop through all options
      cout << setprecision(6) << right << fixed;
      
      cout << "-------------------------------------------------------------------------------------" << endl;
      cout << setw(15) << "beta"
         << setw(14) << "tau1"
         << setw(14) << "tau2"
         << setw(14) << "tau3"
         << setw(14) << "tau2/tau1"
         << setw(14) << "tau3/tau2"
         << endl;
      
      // Define Nsubjettiness functions for beta = 1.0 using one-pass WTA KT axes
      double beta = 1.0;
      Nsubjettiness         nSub1_beta1(1,   OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta));
      Nsubjettiness         nSub2_beta1(2,   OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta));
      Nsubjettiness         nSub3_beta1(3,   OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta));
      NsubjettinessRatio   nSub21_beta1(2,1, OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta));
      NsubjettinessRatio   nSub32_beta1(3,2, OnePass_WTA_KT_Axes(), UnnormalizedMeasure(beta));
      
      // calculate Nsubjettiness values (beta = 1.0)
      double  tau1_beta1 =  nSub1_beta1(this_jet);
      double  tau2_beta1 =  nSub2_beta1(this_jet);
      double  tau3_beta1 =  nSub3_beta1(this_jet);
      double tau21_beta1 = nSub21_beta1(this_jet);
      double tau32_beta1 = nSub32_beta1(this_jet);
      
      // Output results (beta = 1.0)
      cout << setw(15) << 1.0
         << setw(14) << tau1_beta1
         << setw(14) << tau2_beta1
         << setw(14) << tau3_beta1
         << setw(14) << tau21_beta1
         << setw(14) << tau32_beta1
         << endl;
       
        
      outfile << setw(14) << tau1_beta1
         << setw(14) << tau2_beta1
         << setw(14) << tau3_beta1
         << setw(14) << tau21_beta1
         << setw(14) << tau32_beta1
         << endl;
      
      
      
   }
  
}
