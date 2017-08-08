"""
author: DI WU
stevenwudi@gmail.com
"""
import getopt
import sys
import os
import time
import numpy as np
# some configurations files for OBT experiments, originally, I would never do that this way of importing,
# it's simple way too ugly
from scripts.config import SETUP_SEQ, RESULT_SRC, OVERWRITE_RESULT, SAVE_RESULT, SEQ_SRC
from scripts import butil
from scripts.model.result import Result
from keras.preprocessing import image
from scripts.visualisation_utils import plot_tracking_rect, show_precision
OVERWRITE_RESULT = True
DEBUG = True


class Tracker:
    def __init__(self, name=''):
        self.name = name

if OVERWRITE_RESULT:
    from trackers.cvpr_2014_color_name import cvpr_2014_color_name
    from trackers.bmvc_2014_pami_2014_fDSST import bmvc_2014_pami_2014_fDSST


def main(argv):
    if OVERWRITE_RESULT:
        trackers = [cvpr_2014_color_name()]
        trackers = [bmvc_2014_pami_2014_fDSST(number_of_scales=17,
                                              padding=2.0,
                                              interpolate_response=True,
                                              kernel='linear',
                                              compressed_features='gray_hog')]
    else:
        trackers = [Tracker(name='cvpr_2014_color_name')]

    evalTypes = ['OPE']
    loadSeqs = 'TB100'
    try:
        opts, args = getopt.getopt(argv, "ht:e:s:", ["tracker=", "evaltype=", "sequence="])
    except getopt.GetoptError:
        print('usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>')
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print('usage : run_trackers.py -t <trackers> -s <sequences>' + '-e <evaltypes>')
            sys.exit(0)
        elif opt in ("-t", "--tracker"):
            trackers = [x.strip() for x in arg.split(',')]
            # trackers = [arg]
        elif opt in ("-s", "--sequence"):
            loadSeqs = arg
            if loadSeqs != 'All' and loadSeqs != 'all' and \
                            loadSeqs != 'tb50' and loadSeqs != 'tb100' and \
                            loadSeqs != 'cvpr13':
                loadSeqs = [x.strip() for x in arg.split(',')]
        elif opt in ("-e", "--evaltype"):
            evalTypes = [x.strip() for x in arg.split(',')]

    if SETUP_SEQ:
        print('Setup sequences ...')
        butil.setup_seqs(loadSeqs)

    print('Starting benchmark for {0} trackers, evalTypes : {1}'.format(
        len(trackers), evalTypes))
    for evalType in evalTypes:
        seqNames = butil.get_seq_names(loadSeqs)
        seqs = butil.load_seq_configs(seqNames)
        ######################################################################
        trackerResults = run_trackers(trackers, seqs, evalType)
        ######################################################################
        for tracker in trackers:
            results = trackerResults[tracker]
            if len(results) > 0:
                ######################################################################
                evalResults, attrList = butil.calc_result(tracker, seqs, results, evalType, SEQ_SRC)
                ######################################################################
                print ("Result of Sequences\t -- '{0}'".format(tracker.name))
                for i, seq in enumerate(seqs):
                    try:
                        print('\t{0}:\'{1}\'{2}\taveCoverage : {3:.3f}%\taveErrCenter : {4:.3f}'.format(
                            i,
                            seq.name,
                            " " * (12 - len(seq.name)),
                            sum(seq.aveCoverage) / len(seq.aveCoverage) * 100,
                            sum(seq.aveErrCenter) / len(seq.aveErrCenter)))
                    except:
                        print('\t\'{0}\'  ERROR!!'.format(seq.name))

                print("Result of attributes\t -- '{0}'".format(tracker.name))
                for attr in attrList:
                    print("\t\'{}\'\t overlap : {:04.2f}% \t\t failures : {:04.2f}".format(attr.name, attr.overlap, attr.error))

                if SAVE_RESULT:
                    butil.save_scores(attrList)


def run_trackers(trackers, seqs, evalType):
    tmpRes_path = RESULT_SRC.format('tmp/{0}/'.format(evalType))
    if not os.path.exists(tmpRes_path):
        os.makedirs(tmpRes_path)

    numSeq = len(seqs)

    trackerResults = dict((t, list()) for t in trackers)
    ##################################################
    # chose sequence to run from below
    ##################################################
    for idxSeq in range(36, numSeq):
        s = seqs[idxSeq]
        subSeqs, subAnno = butil.get_sub_seqs(s, 20.0, evalType)

        for idxTrk in range(len(trackers)):
            t = trackers[idxTrk]

            if not OVERWRITE_RESULT:

                trk_src = os.path.join(RESULT_SRC.format(evalType), t.name)
                result_src = os.path.join(trk_src, s.name + '.json')
                if os.path.exists(result_src):
                    seqResults = butil.load_seq_result(evalType, t, s.name)
                    trackerResults[t].append(seqResults)
                    continue
            seqResults = []
            seqLen = len(subSeqs)
            for idx in range(seqLen):
                print('{0}_{1}, {2}_{3}:{4}/{5} - {6}'.format(
                    idxTrk + 1, t.name, idxSeq + 1, s.name, idx + 1, seqLen, evalType))
                subS = subSeqs[idx]
                subS.name = s.name + '_' + str(idx)
                ####################
                t, res = run_KCF_variant(t, subS)
                ####################
                r = Result(t.name, s.name, subS.startFrame, subS.endFrame,
                           res['type'], evalType, res['res'], res['fps'], None)
                try:
                    r.tmplsize = res['tmplsize'][0]
                except:
                    pass
                r.refresh_dict()
                seqResults.append(r)
            # end for subseqs
            if SAVE_RESULT:
                butil.save_seq_result(seqResults)

            trackerResults[t].append(seqResults)
            # end for tracker
    # end for allseqs
    return trackerResults


def run_KCF_variant(tracker, seq):

    start_frame = 0
    tracker.res = []
    for frame in range(start_frame, seq.endFrame - seq.startFrame+1):
        image_filename = seq.s_frames[frame]
        image_path = os.path.join(seq.path, image_filename)
        img_rgb = image.load_img(image_path)
        img_rgb = image.img_to_array(img_rgb)

        start_time = time.time()
        if frame == start_frame:
            tracker.train(img_rgb, seq.gtRect[start_frame])
            total_time = 0
        else:
            tracker.detect(img_rgb, frame)
        total_time += time.time() - start_time

        if DEBUG and frame > start_frame:
            print("Frame ==", frame)
            print('horiz_delta: %.2f, vert_delta: %.2f' % (tracker.horiz_delta, tracker.vert_delta))
            print("pos", np.array(tracker.res[-1]).astype(int))
            print("gt", np.array(seq.gtRect[frame]).astype(int))
            print("\n")
            plot_tracking_rect(seq.name, frame + seq.startFrame, img_rgb, tracker, seq.gtRect)

    tracker.fps = len(tracker.res) / total_time
    print("Frames-per-second:", tracker.fps)

    if DEBUG:
        tracker.precisions = show_precision(np.array(tracker.res), np.array(seq.gtRect), seq.name)

    res = {'type': 'rect', 'res': tracker.res, 'fps': tracker.fps}

    return tracker, res

if __name__ == "__main__":
    main(sys.argv[1:])
