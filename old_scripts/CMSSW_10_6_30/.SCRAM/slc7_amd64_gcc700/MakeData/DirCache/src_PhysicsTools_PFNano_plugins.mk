ifeq ($(strip $(PhysicsToolsNanoAODJMARPlugins)),)
PhysicsToolsNanoAODJMARPlugins := self/src/PhysicsTools/PFNano/plugins
PLUGINS:=yes
PhysicsToolsNanoAODJMARPlugins_files := $(patsubst src/PhysicsTools/PFNano/plugins/%,%,$(foreach file,*.cc,$(eval xfile:=$(wildcard src/PhysicsTools/PFNano/plugins/$(file)))$(if $(xfile),$(xfile),$(warning No such file exists: src/PhysicsTools/PFNano/plugins/$(file). Please fix src/PhysicsTools/PFNano/plugins/BuildFile.))))
PhysicsToolsNanoAODJMARPlugins_BuildFile    := $(WORKINGDIR)/cache/bf/src/PhysicsTools/PFNano/plugins/BuildFile
PhysicsToolsNanoAODJMARPlugins_LOC_USE := self  FWCore/Framework FWCore/MessageLogger DataFormats/PatCandidates DataFormats/VertexReco DataFormats/NanoAOD CommonTools/Utils CommonTools/UtilAlgos CommonTools/CandAlgos RecoBTag/FeatureTools RecoJets/JetAlgorithms TrackingTools/Records roottmva fastjet fastjet-contrib
PhysicsToolsNanoAODJMARPlugins_PRE_INIT_FUNC += $$(eval $$(call edmPlugin,PhysicsToolsNanoAODJMARPlugins,PhysicsToolsNanoAODJMARPlugins,$(SCRAMSTORENAME_LIB),src/PhysicsTools/PFNano/plugins))
PhysicsToolsNanoAODJMARPlugins_PACKAGE := self/src/PhysicsTools/PFNano/plugins
ALL_PRODS += PhysicsToolsNanoAODJMARPlugins
PhysicsTools/PFNano_forbigobj+=PhysicsToolsNanoAODJMARPlugins
PhysicsToolsNanoAODJMARPlugins_INIT_FUNC        += $$(eval $$(call Library,PhysicsToolsNanoAODJMARPlugins,src/PhysicsTools/PFNano/plugins,src_PhysicsTools_PFNano_plugins,$(SCRAMSTORENAME_BIN),,$(SCRAMSTORENAME_LIB),$(SCRAMSTORENAME_LOGS),edm))
PhysicsToolsNanoAODJMARPlugins_CLASS := LIBRARY
else
$(eval $(call MultipleWarningMsg,PhysicsToolsNanoAODJMARPlugins,src/PhysicsTools/PFNano/plugins))
endif
ALL_COMMONRULES += src_PhysicsTools_PFNano_plugins
src_PhysicsTools_PFNano_plugins_parent := PhysicsTools/PFNano
src_PhysicsTools_PFNano_plugins_INIT_FUNC += $$(eval $$(call CommonProductRules,src_PhysicsTools_PFNano_plugins,src/PhysicsTools/PFNano/plugins,PLUGINS))
