from pydantic import BaseModel

class CandidateData(BaseModel):
    age: int
    accessibility: int
    education: int
    employment: int
    gender: int
    mental_health: int
    main_branch: int
    years_code: int
    years_code_pro: int
    salary: float
    num_skills: int 
    continent: int
