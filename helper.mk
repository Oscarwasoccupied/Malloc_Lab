SUBMIT_MESSAGE =
SUBMIT_MESSAGE += "I affirm that I have complied with this course's academic"
SUBMIT_MESSAGE += "integrity policy as defined at"
SUBMIT_MESSAGE += "https://www.cs.cmu.edu/~213/academicintegrity.html [y/N]: "

SELECT_COURSE_MESSAGE =
SELECT_COURSE_MESSAGE += "\nSelect the course you are enrolled in.\n"
SELECT_COURSE_MESSAGE += "\nUse the arrow keys and Enter to select an option."

# Set course/lab variables
include .labname.mk

#####################################################################
# Validate variables
#####################################################################

ifndef COURSECODE
  $(error COURSECODE is not set)
endif
ifndef HANDIN_FILES
  $(error HANDIN_FILES is not set)
endif
ifndef FORMAT_FILES
  $(error FORMAT_FILES is not set)
endif

# Detect if we should submit separately
ifdef LAB_CHECKPOINT
  ifdef LAB_FINAL
    SUBMIT_SEPARATE = 1
  else
    $(error LAB_CHECKPOINT set, but not LAB_FINAL)
  endif
else
  ifdef LAB_FINAL
    $(error LAB_FINAL set, but not LAB_CHECKPOINT)
  endif
  ifndef LAB
    $(error LAB is not set)
  endif
endif

# Check handin parameters
ifdef HANDIN_TAR
  HANDIN_FILE = $(HANDIN_TAR)
else
  ifneq ($(words $(HANDIN_FILES)),1)
    $(error HANDIN_TAR not set, and HANDIN_FILES has more than one file)
  endif
  HANDIN_FILE = $(HANDIN_FILES)
endif

#####################################################################
# Rules to create and verify handin files
#####################################################################

ifdef HANDIN_TAR

$(HANDIN_TAR): $(HANDIN_FILES)
	tar cvf $@ $^

.PHONY: handin
handin: $(HANDIN_TAR)

else

.PHONY: handin
handin:
	@echo "There is no handin tar. Submit \"$(HANDIN_FILE)' directly."

endif

#####################################################################
# Rules to submit to Autolab
#####################################################################

ifneq ($(words $(COURSECODE)),1)

submit submit-checkpoint submit-final: | .selected_course.txt submit-check-selected

.selected_course.txt:
	@MESSAGE=$$(whiptail --title "Autolab submission" \
	    --menu "$$(echo $(SELECT_COURSE_MESSAGE))" \
	           14 70 $(words $(COURSECODE)) \
	    $(foreach code,$(COURSECODE),"$(code)" "") \
	    3>&1 1>&2 2>&3) \
	  && echo $$MESSAGE > $@

SELECTED_COURSECODE = $(shell cat .selected_course.txt)

.PHONY: submit-check-selected
submit-check-selected:
	@[ $(words $(filter $(SELECTED_COURSECODE),$(COURSECODE))) -eq 1 ] || ( \
	  echo "ERROR: Selected course \"$(SELECTED_COURSECODE)\" is invalid."; \
	  echo "       Please run \"make select-course\" again."; \
	  exit 1)
	@echo "Selected course:  $(SELECTED_COURSECODE) " \
	  "(\"make select-course\" to change)"

.PHONY: select-course
select-course:
	@rm -rf .selected_course.txt
	@make -s .selected_course.txt

else

SELECTED_COURSECODE = $(COURSECODE)

endif

ifdef SUBMIT_SEPARATE

.PHONY: submit
submit:
	@echo "Please use one of the following:"
	@echo "  make submit-checkpoint"
	@echo "  make submit-final"
	@exit 1

.PHONY: submit-checkpoint
submit-checkpoint: $(HANDIN_FILE)
	@echo
	@echo "================================================="
	@echo "Submitting to CHECKPOINT ($(LAB_CHECKPOINT))"
	@echo "================================================="
	@echo
	@echo -n $(SUBMIT_MESSAGE) | fold -s -w 76
	@read -n 1 && echo && echo $$REPLY | grep -iq "^y"
	@autolab submit $(SELECTED_COURSECODE):$(LAB_CHECKPOINT) $<

.PHONY: submit-final
submit-final: $(HANDIN_FILE)
	@echo
	@echo "================================================="
	@echo "Submitting to FINAL ($(LAB_FINAL))"
	@echo "================================================="
	@echo
	@echo -n $(SUBMIT_MESSAGE) | fold -s -w 76
	@read -n 1 && echo && echo $$REPLY | grep -iq "^y"
	@autolab submit $(SELECTED_COURSECODE):$(LAB_FINAL) $<

else

.PHONY: submit
submit: $(HANDIN_FILE)
	@echo -n $(SUBMIT_MESSAGE) | fold -s -w 76
	@read -n 1 && echo && echo $$REPLY | grep -iq "^y"
	@autolab submit $(SELECTED_COURSECODE):$(LAB) $<

endif

#####################################################################
# Rules to check code formatting
#####################################################################

.PHONY: format
format: $(FORMAT_FILES)
	$(LLVM_PATH)clang-format -style=file -i $^

.PHONY: check-format
check-format: $(FORMAT_FILES)
	@! LC_ALL=C grep -P -n "[\x00-\x08\x0E-\x1F\x80-\xFF]" $^ || ( \
	  echo; \
	  echo "ERROR: Your code contains non-ASCII characters!"; \
	  echo "    The file names and line numbers are shown above."; \
	  echo "    Please remove these characters before continuing."; \
	  echo; \
	  echo "    Files checked: $^"; \
	  echo; \
	  echo "    You must fix this before submitting to Autolab."; \
	  echo; \
	  $(if $(FORCE_FORMAT),exit 1,sleep 1))
	@$(LLVM_PATH)clang-format -style=file $^ | cmp --silent <(cat $^) - || ( \
	  echo; \
	  echo "ERROR: Your code's formatting does not match clang-format."; \
	  echo "    For details, see https://www.cs.cmu.edu/~213/codeStyle.html"; \
	  echo "    To reformat your code, run \"make format\"."; \
	  echo; \
	  echo "    Files checked: $^"; \
	  echo; \
	  echo "    You must fix this before submitting to Autolab."; \
	  echo; \
	  $(if $(FORCE_FORMAT),exit 1,sleep 1))
