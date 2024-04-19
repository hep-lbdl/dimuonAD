// -*- C++ -*-


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//class to extract relevant info
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include <vector>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

string output_file_path = "test.dat";
ofstream outfile;


//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class MiniAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit MiniAnalyzer(const edm::ParameterSet&);
      ~MiniAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------

     edm::EDGetTokenT<pat::MuonCollection> muonToken_;
     edm::EDGetTokenT<pat::JetCollection> jetToken_;
     edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MiniAnalyzer::MiniAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
   usesResource("TFileService");

   muonToken_ = consumes<pat::MuonCollection>(edm::InputTag("slimmedMuons"));
   jetToken_ = consumes<pat::JetCollection>(edm::InputTag("slimmedJets"));
   vtxToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));


   
   outfile.open(output_file_path);      
   outfile << setprecision(16) << right << fixed;

    

}


MiniAnalyzer::~MiniAnalyzer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

   outfile.close();

}

//
// member functions
//

// ------------ method called for each event  ------------
void
MiniAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


   //define the handler, a token and get the information by token
   Handle<pat::MuonCollection> muons;
   iEvent.getByToken(muonToken_, muons);

   Handle<pat::JetCollection> jets;
   iEvent.getByToken(jetToken_, jets); 
    
   Handle<reco::VertexCollection> vertices;
   iEvent.getByToken(vtxToken_, vertices);
   if (vertices->empty()) return; // skip the event if no PV is found
   const reco::Vertex &PV = vertices->front();

   // Define variables for muon, amuon
   float muon_pt, muon_eta, muon_phi, muon_iso03, muon_iso04;
   float amuon_pt, amuon_eta, amuon_phi, amuon_iso03, amuon_iso04;
   bool muon_isTight;
   bool amuon_isTight;
   bool found_muon, found_amuon;


   found_muon = false;
   found_amuon = false;

   //if collection is valid, loop over muons in event
   if(muons.isValid()){
      for (const pat::Muon &itmuon : *muons){

        if ((itmuon.charge() == -1) && (!found_muon)) {
            found_muon = true;
            muon_pt = itmuon.pt();
            muon_eta = itmuon.eta();
            muon_phi = itmuon.phi();
            muon_isTight = itmuon.isTightMuon(PV);
            
            if (itmuon.isPFMuon() && itmuon.isPFIsolationValid()) {
                auto iso03 = itmuon.pfIsolationR03();
                muon_iso03 = (iso03.sumChargedHadronPt + iso03.sumNeutralHadronEt + iso03.sumPhotonEt)/itmuon.pt();
                auto iso04 = itmuon.pfIsolationR04();
                muon_iso04 = (iso04.sumChargedHadronPt + iso04.sumNeutralHadronEt + iso04.sumPhotonEt)/itmuon.pt();
            }
            else {
                muon_iso03 = -999;
                muon_iso04 = -999;   
            }
        }

        if ((itmuon.charge() == 1) && (!found_amuon)) {
            found_amuon = true;
            amuon_pt = itmuon.pt();
            amuon_eta = itmuon.eta();
            amuon_phi = itmuon.phi();
            amuon_isTight = itmuon.isTightMuon(PV);

            if (itmuon.isPFMuon() && itmuon.isPFIsolationValid()) {
                auto iso03 = itmuon.pfIsolationR03();
                amuon_iso03 = (iso03.sumChargedHadronPt + iso03.sumNeutralHadronEt + iso03.sumPhotonEt)/itmuon.pt();
                auto iso04 = itmuon.pfIsolationR04();
                amuon_iso04 = (iso04.sumChargedHadronPt + iso04.sumNeutralHadronEt + iso04.sumPhotonEt)/itmuon.pt();
            }
            else {
                amuon_iso03 = -999;
                amuon_iso04 = -999;        
            }
        }
      }
   }
    
   // If both a muon and an antimuon have been found, write out the event
   if (found_muon && found_amuon) { 
       if (muon_isTight && amuon_isTight) { 
           outfile << setw(30) << muon_pt
                   << setw(30) << muon_eta
                   << setw(30) << muon_phi
                   << setw(30) << muon_iso03
                   << setw(30) << muon_iso04
                   << setw(30) << amuon_pt
                   << setw(30) << amuon_eta
                   << setw(30) << amuon_phi
                   << setw(30) << amuon_iso03
                   << setw(30) << amuon_iso04
                   << endl;

   }

}

}


// ------------ method called once each job just before starting event loop  ------------
void
MiniAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
MiniAnalyzer::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MiniAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MiniAnalyzer);
