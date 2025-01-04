ALL_PACKAGES += Test/MiniAnalyzer
subdirs_src_Test_MiniAnalyzer := src_Test_MiniAnalyzer_python src_Test_MiniAnalyzer_plugins
ALL_SUBSYSTEMS+=Test
subdirs_src_Test = src_Test_MiniAnalyzer
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
ALL_PACKAGES += PhysicsTools/PFNano
subdirs_src_PhysicsTools_PFNano := src_PhysicsTools_PFNano_plugins src_PhysicsTools_PFNano_python
ALL_SUBSYSTEMS+=PhysicsTools
subdirs_src_PhysicsTools = src_PhysicsTools_PFNano
ifeq ($(strip $(PyPhysicsToolsPFNano)),)
PyPhysicsToolsPFNano := self/src/PhysicsTools/PFNano/python
src_PhysicsTools_PFNano_python_parent := 
ALL_PYTHON_DIRS += $(patsubst src/%,%,src/PhysicsTools/PFNano/python)
PyPhysicsToolsPFNano_files := $(patsubst src/PhysicsTools/PFNano/python/%,%,$(wildcard $(foreach dir,src/PhysicsTools/PFNano/python ,$(foreach ext,$(SRC_FILES_SUFFIXES),$(dir)/*.$(ext)))))
PyPhysicsToolsPFNano_LOC_USE := self  
PyPhysicsToolsPFNano_PACKAGE := self/src/PhysicsTools/PFNano/python
ALL_PRODS += PyPhysicsToolsPFNano
PyPhysicsToolsPFNano_INIT_FUNC        += $$(eval $$(call PythonProduct,PyPhysicsToolsPFNano,src/PhysicsTools/PFNano/python,src_PhysicsTools_PFNano_python,1,1,$(SCRAMSTORENAME_PYTHON),$(SCRAMSTORENAME_LIB),,))
else
$(eval $(call MultipleWarningMsg,PyPhysicsToolsPFNano,src/PhysicsTools/PFNano/python))
endif
ALL_COMMONRULES += src_PhysicsTools_PFNano_python
src_PhysicsTools_PFNano_python_INIT_FUNC += $$(eval $$(call CommonProductRules,src_PhysicsTools_PFNano_python,src/PhysicsTools/PFNano/python,PYTHON))
