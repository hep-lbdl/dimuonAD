ifeq ($(strip $(PhysicsTools/PFNano)),)
src_PhysicsTools_PFNano := self/PhysicsTools/PFNano
PhysicsTools/PFNano  := src_PhysicsTools_PFNano
src_PhysicsTools_PFNano_BuildFile    := $(WORKINGDIR)/cache/bf/src/PhysicsTools/PFNano/BuildFile
src_PhysicsTools_PFNano_LOC_USE := DataFormats/NanoAOD boost DataFormats/Common self CommonTools/UtilAlgos DataFormats/Candidate DataFormats/StdDictionaries CommonTools/Utils FWCore/Common CommonTools/CandAlgos DataFormats/PatCandidates FWCore/Utilities
src_PhysicsTools_PFNano_EX_USE   := $(foreach d,$(src_PhysicsTools_PFNano_LOC_USE),$(if $($(d)_EX_FLAGS_NO_RECURSIVE_EXPORT),,$d))
ALL_EXTERNAL_PRODS += src_PhysicsTools_PFNano
src_PhysicsTools_PFNano_INIT_FUNC += $$(eval $$(call EmptyPackage,src_PhysicsTools_PFNano,src/PhysicsTools/PFNano))
endif

