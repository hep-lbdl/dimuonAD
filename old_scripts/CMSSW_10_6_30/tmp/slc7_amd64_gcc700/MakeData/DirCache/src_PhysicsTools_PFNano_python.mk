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
