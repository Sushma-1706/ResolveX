from __future__ import annotations
def score_sets(truth, predicted):
    values=[]; tp=fp=fn=single_correct=single_total=0
    for sid, actual in truth.items():
        a,p=set(actual),set(predicted.get(sid,[])); tp_i=len(a&p); fp_i=len(p-a); fn_i=len(a-p)
        if not a:
            values.append(1.0 if not p else 0.0); single_total+=1; single_correct+=int(not p)
        else:
            precision=tp_i/(tp_i+fp_i) if tp_i+fp_i else 0.; recall=tp_i/(tp_i+fn_i) if tp_i+fn_i else 0.
            values.append(1.25*precision*recall/(.25*precision+recall) if precision+recall else 0.)
        tp+=tp_i;fp+=fp_i;fn+=fn_i
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {"f0_5":sum(values)/len(values) if values else 0.,"precision":precision,"recall":recall,"false_positives":fp,"false_negatives":fn,"singleton_accuracy":single_correct/single_total if single_total else 0.}
