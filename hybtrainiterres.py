import numpy as np

def hybtrainiterres(TrainRes, witer, fvaliter, projhyb):
    TrainRes['iter'] += 1
    TrainRes['witer'][TrainRes['iter'], :projhyb['mlm']['nw']] = witer
    TrainRes['sjctrain'][TrainRes['iter']] = 0
    TrainRes['sjcval'][TrainRes['iter']] = 0
    TrainRes['sjctest'][TrainRes['iter']] = 0
    TrainRes['sjrtrain'][TrainRes['iter']] = 0
    TrainRes['sjrval'][TrainRes['iter']] = 0
    TrainRes['sjrtest'][TrainRes['iter']] = 0
    TrainRes['AICc'][TrainRes['iter']] = 0
    cnt_jctrain = cnt_jcval = cnt_jctest = 0
    
    for i in range(projhyb['nbatch']):
        nres, _, jctr, _, jcvl, _, jrtr, _, jrvl = hybbatcherrors(
            projhyb, witer, projhyb['batch'][i])
        
        if projhyb['istrain'][i] == 1:  # Training batch
            TrainRes['sjctrain'][TrainRes['iter']] += jctr
            TrainRes['sjcval'][TrainRes['iter']] += jcvl
            TrainRes['sjrtrain'][TrainRes['iter']] += jrtr
            TrainRes['sjrval'][TrainRes['iter']] += jrvl
            cnt_jctrain += nres
        elif projhyb['istrain'][i] == 3:  # Testing batch
            TrainRes['sjctest'][TrainRes['iter']] += jctr
            TrainRes['sjrtest'][TrainRes['iter']] += jrtr
            cnt_jctest += nres
    
    cnt_jctrain = max(cnt_jctrain, 1)
    cnt_jctest = max(cnt_jctest, 1)
    
    for field in ['sjctrain', 'sjcval', 'sjctest', 'sjrtrain', 'sjrval', 'sjrtest']:
        if 'train' in field or 'val' in field:
            TrainRes[field][TrainRes['iter']] /= cnt_jctrain
        else:
            TrainRes[field][TrainRes['iter']] /= cnt_jctest
    
    TrainRes['AICc'][TrainRes['iter']] = cnt_jctrain * np.log(TrainRes['sjctrain'][TrainRes['iter']]) + \
        2 * projhyb['mlm']['nw'] + \
        2 * projhyb['mlm']['nw'] * (projhyb['mlm']['nw'] + 1) / (cnt_jctrain - projhyb['mlm']['nw'] - 1)
    
    TrainRes['resnorm'][TrainRes['iter']] = fvaliter
    print(f"{TrainRes['iter']} {TrainRes['resnorm'][TrainRes['iter']]:.2E} {TrainRes['sjctrain'][TrainRes['iter']]:.2E} {TrainRes['sjcval'][TrainRes['iter']]:.2E} {TrainRes['sjctest'][TrainRes['iter']]:.2E} {TrainRes['sjrtrain'][TrainRes['iter']]:.2E} {TrainRes['sjrval'][TrainRes['iter']]:.2E} {TrainRes['sjrtest'][TrainRes['iter']]:.2E} {TrainRes['AICc'][TrainRes['iter']]:.2E} {projhyb['mlm']['nw']} {time.process_time() - TrainRes['t0']:.2E}")
    
    return TrainRes