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
