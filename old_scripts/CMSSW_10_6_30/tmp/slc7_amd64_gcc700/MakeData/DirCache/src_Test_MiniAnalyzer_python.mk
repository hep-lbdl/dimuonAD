ifeq ($(strip $(PyTestMiniAnalyzer)),)
PyTestMiniAnalyzer := self/src/Test/MiniAnalyzer/python
src_Test_MiniAnalyzer_python_parent := 
ALL_PYTHON_DIRS += $(patsubst src/%,%,src/Test/MiniAnalyzer/python)
PyTestMiniAnalyzer_files := $(patsubst src/Test/MiniAnalyzer/python/%,%,$(wildcard $(foreach dir,src/Test/MiniAnalyzer/python ,$(foreach ext,$(SRC_FILES_SUFFIXES),$(dir)/*.$(ext)))))
PyTestMiniAnalyzer_LOC_USE := self  
PyTestMiniAnalyzer_PACKAGE := self/src/Test/MiniAnalyzer/python
ALL_PRODS += PyTestMiniAnalyzer
PyTestMiniAnalyzer_INIT_FUNC        += $$(eval $$(call PythonProduct,PyTestMiniAnalyzer,src/Test/MiniAnalyzer/python,src_Test_MiniAnalyzer_python,1,1,$(SCRAMSTORENAME_PYTHON),$(SCRAMSTORENAME_LIB),,))
else
$(eval $(call MultipleWarningMsg,PyTestMiniAnalyzer,src/Test/MiniAnalyzer/python))
endif
ALL_COMMONRULES += src_Test_MiniAnalyzer_python
src_Test_MiniAnalyzer_python_INIT_FUNC += $$(eval $$(call CommonProductRules,src_Test_MiniAnalyzer_python,src/Test/MiniAnalyzer/python,PYTHON))
