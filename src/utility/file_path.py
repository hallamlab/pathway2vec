import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split('/')
REPO_PATH = '/'.join(REPO_PATH[:-2])

FUNCTIONAL_PATH = os.path.join(REPO_PATH, 'database', 'functional')
COG_PATH = os.path.join(FUNCTIONAL_PATH, 'COG_2013-12-27')
KEGG_PATH = os.path.join(FUNCTIONAL_PATH, 'kegg-pep-2011-06-18')
METACYC_PATH = os.path.join(FUNCTIONAL_PATH, 'metacyc-v5-2011-10-21')
SEED_PATH = os.path.join(FUNCTIONAL_PATH, 'CAZY_2014_09_04')
CAZY_PATH = os.path.join(FUNCTIONAL_PATH, 'CAZY_2014_09_04')
REFSEQ_PATH = os.path.join(FUNCTIONAL_PATH, 'refseq-nr-2014-01-18_NoWrap_1')

DATABASE_PATH = os.path.join(REPO_PATH, 'database', 'biocyc-flatfiles')
LOG_PATH = os.path.join(REPO_PATH, 'log')
OBJECT_PATH = os.path.join(REPO_PATH, 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'dataset')
INPUT_PATH = os.path.join(REPO_PATH, 'inputset')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
GUTCYC_PATH = os.path.join(REPO_PATH, 'GUTCYC_FREE_OUTPUT')
MODEL_PATH = os.path.join(REPO_PATH, 'model')
FEATURE_PATH = os.path.join(DIRECTORY_PATH, 'feature_builder')

GLPK_PATH = os.path.join(REPO_PATH, 'model', 'glpk-4.63/examples/glpsol')
PTOOLS_PATH = os.path.join('/'.join(DIRECTORY_PATH.split('/')[:3]), 'pathway-tools/pathway-tools')
