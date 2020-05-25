import SimpleITK as sitk

import pymia.evaluation.metric as mt
import pymia.evaluation.evaluator as ev

img_ref = sitk.ReadImage('/home/fbalsiger/PyCharmProjects/pymia/examples/dummy-data/Subject_1/Subject_1_GTN.mha')
img_pred = sitk.Image(*img_ref.GetSize(), img_ref.GetPixelID())
img_pred.CopyInformation(img_ref)

evaluator = ev.Evaluator(ev.ConsoleEvaluatorWriter())
evaluator.add_label(1, 'TEST')
evaluator.metrics = mt.get_all_segmentation_metrics()

evaluator.evaluate(img_pred, img_pred, 'Test')