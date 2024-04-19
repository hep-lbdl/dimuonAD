import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'root://eospublic.cern.ch//eos/opendata/cms/mc/RunIISummer20UL16MiniAODv2/ggXToYY_YToMuMu_M22p5_JPCZeroPlusPlus_TuneCP5_13TeV-pythia8-JHUGen/MINIAODSIM/106X_mcRun2_asymptotic_v17-v2/2430000/42881FC8-5808-BC4B-BE75-DC586C2AEFB5.root'
    )
)

process.demo = cms.EDAnalyzer('MiniAnalyzer',
                              muons = cms.InputTag("slimmedMuons"),
                              jets = cms.InputTag("slimmedJets"),
                              vertices = cms.InputTag("offlineSlimmedPrimaryVertices")
)


process.p = cms.Path(process.demo)
