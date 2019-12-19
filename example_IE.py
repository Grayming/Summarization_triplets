#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:28:57 2019

@author: mingliu
"""

import IE
import timeit
import numpy as np

extractor = IE.IE_extraction()

s='CT Sinuses performed on 23-DEC-2008 at 01:30 PM:    Technique:  Unenhanced multi detector CT images of the paranasal sinuses with coronal and sagittal reformats.      Findings:  Comparison with similar study dated 17 September 2008.  Minor mucoperiosteal thickening in the anterior and posterior ethmoid air cells bilaterally and the left frontal recess.  The maxillary sinuses, right frontal, and both sphenoid sinuses are clear.  No bony sclerosis or erosion.  The osteomeatal complexes are patent bilaterally.    Impression:  Minor mucoperiosteal thickening in the ethmoid and left frontal sinuses.  No acute or chronic sinusitis.       Co-signed Dr M Pianta. CT Sinuses performed on 30-MAR-2009 at 09:28 AM:    CT scan of the para nasal sinuses.    History:    Exclude fungal disease.  Chemotherapy for ALL.    Procedure:    Helical scans were performed through the para nasal sinuses with sagittal and coronal reformat images.    Findings:    There is a trace of mucosal thickening seen in the dependent portion of both maxillary antra.  The osteomeatal complex is patent bilaterally.  The remaining para nasal sinuses are clear.  No evidence of erosion of the sinus walls.  The nasal cavity is unremarkable.    Comment:    No significant sinus pathology identified.  Specifically no evidence of fungal sinusitis.'

start = timeit.default_timer()
triplets = extractor.final_triplets(s)
stop = timeit.default_timer()
print('Time: ', stop - start)
