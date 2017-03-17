


# Rules for CRF-based run 2 

# Directory with all tsv files
CVDATA_IN?=cv.d

# Fixed number of epochs 35, no dev set needed
# Note: with a dev set, the numbers don't change much.

EPOCHS?=35

# Set minimal sequence length (0 takes all sequences)
MINLEN?=0

# 
TRAINMINLEN?=0

# Number of threads for wapiti training
THREADS?=4

CVDATA_OUT=run2_out_$(EPOCHS)_$(MINLEN).d

FOLDS:= _0 _1 _2 _3 _4 _5 _6 _7 _8 _9



cvdata-out-files:=$(foreach f,$(FOLDS), $(CVDATA_OUT)/train$f.tsv)
cvdata-out-files:=$(foreach f,$(FOLDS), $(CVDATA_OUT)/test$f.tsv)
cvdata-out-files+=$(foreach f,$(FOLDS), $(CVDATA_OUT)/train$f.mod)
cvdata-out-files+=$(foreach f,$(FOLDS), $(CVDATA_OUT)/test$f.eval.tsv)


cvdata-out-files+=$(CVDATA_OUT)/test_X.eval.tsv

cvdata-out-files+=$(CVDATA_OUT)/$(TRAINMINLEN)_$(MINLEN).mean.std.tsv



target: $(cvdata-out-files)


$(CVDATA_OUT)/train%.tsv $(CVDATA_OUT)/test%.tsv : $(CVDATA_IN)/train%.tsv $(CVDATA_IN)/test%.tsv
	mkdir -p $(CVDATA_OUT)
	python3 lib/gdi2wapiti.py -w -L $(MINLEN) < $(CVDATA_IN)/test$*.tsv > $(CVDATA_OUT)/test$*.tsv
	python3 lib/gdi2wapiti.py -w -L $(TRAINMINLEN) < $(CVDATA_IN)/train$*.tsv > $(CVDATA_OUT)/train$*.tsv




$(CVDATA_OUT)/test%.eval.tsv $(CVDATA_OUT)/test%.eval.tsv.err: $(CVDATA_OUT)/train%.mod  $(CVDATA_OUT)/test%.tsv
	wapiti label -p -s -c -m $< $(word 2,$+) > $@ 2> >(tee $@.err >&2) 
	wapiti label -p -l -c -m $< $(word 2,$+) | python lib/wapitilabels2gdi.py > $(@:.eval.tsv=.prediction.tsv)

$(CVDATA_OUT)/train%.mod: $(CVDATA_OUT)/train%.tsv
	wapiti  train -i $(EPOCHS) -t $(THREADS)  $<  > $@  2> >(tee $@.err >&2)
	
$(CVDATA_OUT)/$(TRAINMINLEN)_$(MINLEN).mean.std.tsv: $(foreach f, $(FOLDS), $(CVDATA_OUT)/test$f.eval.tsv.err)
	python lib/wapiti_cv_eval.py $+ > $@ && cat $@ || rm -f $@
	
%.d:
	mkdir -p $@
$(CVDATA_IN)/test_X.tsv: $(CVDATA_IN)/test.tsv
	cd $(CVDATA_IN)/ && ln -sf $(<F) $(@F)
$(CVDATA_IN)/train_X.tsv: $(CVDATA_IN)/train.tsv
	cd $(CVDATA_IN)/ && ln -sf $(<F) $(@F)


SHELL:=/bin/bash
export SHELLOPTS:=errexit:pipefail
MAKEFLAGS += --no-builtin-rules
.SECONDARY:
