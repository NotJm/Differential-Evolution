from functions.cec2017problems_update import *
SIZE_POPULATION = 100
GENERATIONS = 3000
EXECUTIONS = 50
DIRECTORY = "mnt/data/cec2024"
CURRENT_PROBLEM = "C27"
PROBLEM_PREFIX = "CEC2024"
BASELINE = "AFC"
PROPOSAL = [ "beta", "boundary", "evolutionary", "reflection", "res&rand"]
DIRECTORY_BASELINE = f"{DIRECTORY}/{BASELINE}/{PROBLEM_PREFIX}_{CURRENT_PROBLEM}_{BASELINE}.csv"
DIRECTORY_PROPOSAL = f"{DIRECTORY}/{PROPOSAL}/{PROBLEM_PREFIX}_{CURRENT_PROBLEM}_{PROPOSAL}.csv"

BCHM = [
    "AFC",
    "beta",
    "res&rand",
    "evolutionary",
    "reflection",
    "boundary"
]

EXCLUDE = [
    # "boundary",
    # "evolutionary",
    # "reflection",
    # "res&rand",
]
    
PROBLEMS = {
    # "C01": CEC2017_C01,
    # "C02": CEC2017_C02,
    # "C03": CEC2017_C03,
    # "C04": CEC2017_C04,
    # "C05": CEC2017_C05,
    # "C06": CEC2017_C06,
    # "C07": CEC2017_C07,
    # "C08": CEC2017_C08,
    # "C09": CEC2017_C09,
    # "C10": CEC2017_C10,
    # "C11": CEC2017_C11,
    # "C12": CEC2017_C12,
    # "C13": CEC2017_C13,
    # "C14": CEC2017_C14,
    # "C15": CEC2017_C15,
    # "C16": CEC2017_C16,
    # "C17": CEC2017_C17,
    # "C18": CEC2017_C18,
    # "C19": CEC2017_C19,
    # "C20": CEC2017_C20,
    # "C21": CEC2017_C21,
    # "C22": CEC2017_C22,
    # "C23": CEC2017_C23,
    # "C24": CEC2017_C24,
    # "C25": CEC2017_C25,
    # "C26": CEC2017_C26,
    "C27": CEC2017_C27,
    # "C28": CEC2017_C28,
}
