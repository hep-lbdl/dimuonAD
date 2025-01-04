ifeq ($(strip $(TestMiniAnalyzerAuto)),)
TestMiniAnalyzerAuto := self/src/Test/MiniAnalyzer/plugins
PLUGINS:=yes
TestMiniAnalyzerAuto_files := $(patsubst src/Test/MiniAnalyzer/plugins/%,%,$(wildcard $(foreach dir,src/Test/MiniAnalyzer/plugins ,$(foreach ext,$(SRC_FILES_SUFFIXES),$(dir)/*.$(ext)))))
TestMiniAnalyzerAuto_BuildFile    := $(WORKINGDIR)/cache/bf/src/Test/MiniAnalyzer/plugins/BuildFile
TestMiniAnalyzerAuto_LOC_USE := self  FWCore/Framework FWCore/PluginManager FWCore/ParameterSet DataFormats/PatCandidates
TestMiniAnalyzerAuto_PRE_INIT_FUNC += $$(eval $$(call edmPlugin,TestMiniAnalyzerAuto,TestMiniAnalyzerAuto,$(SCRAMSTORENAME_LIB),src/Test/MiniAnalyzer/plugins))
TestMiniAnalyzerAuto_PACKAGE := self/src/Test/MiniAnalyzer/plugins
ALL_PRODS += TestMiniAnalyzerAuto
Test/MiniAnalyzer_forbigobj+=TestMiniAnalyzerAuto
TestMiniAnalyzerAuto_INIT_FUNC        += $$(eval $$(call Library,TestMiniAnalyzerAuto,src/Test/MiniAnalyzer/plugins,src_Test_MiniAnalyzer_plugins,$(SCRAMSTORENAME_BIN),,$(SCRAMSTORENAME_LIB),$(SCRAMSTORENAME_LOGS),edm))
TestMiniAnalyzerAuto_CLASS := LIBRARY
else
$(eval $(call MultipleWarningMsg,TestMiniAnalyzerAuto,src/Test/MiniAnalyzer/plugins))
endif
ALL_COMMONRULES += src_Test_MiniAnalyzer_plugins
src_Test_MiniAnalyzer_plugins_parent := Test/MiniAnalyzer
src_Test_MiniAnalyzer_plugins_INIT_FUNC += $$(eval $$(call CommonProductRules,src_Test_MiniAnalyzer_plugins,src/Test/MiniAnalyzer/plugins,PLUGINS))
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
ifeq ($(strip $(PhysicsTools/PFNano)),)
src_PhysicsTools_PFNano := self/PhysicsTools/PFNano
PhysicsTools/PFNano  := src_PhysicsTools_PFNano
src_PhysicsTools_PFNano_BuildFile    := $(WORKINGDIR)/cache/bf/src/PhysicsTools/PFNano/BuildFile
src_PhysicsTools_PFNano_LOC_USE := DataFormats/NanoAOD boost DataFormats/Common self CommonTools/UtilAlgos DataFormats/Candidate DataFormats/StdDictionaries CommonTools/Utils FWCore/Common CommonTools/CandAlgos DataFormats/PatCandidates FWCore/Utilities
src_PhysicsTools_PFNano_EX_USE   := $(foreach d,$(src_PhysicsTools_PFNano_LOC_USE),$(if $($(d)_EX_FLAGS_NO_RECURSIVE_EXPORT),,$d))
ALL_EXTERNAL_PRODS += src_PhysicsTools_PFNano
src_PhysicsTools_PFNano_INIT_FUNC += $$(eval $$(call EmptyPackage,src_PhysicsTools_PFNano,src/PhysicsTools/PFNano))
endif

