#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pydantic import BaseModel


class CriterionResult(BaseModel):
    criterion: str
    achieved: bool
    marks_awarded: float
    explanation: str


class EvaluationResult(BaseModel):
    criteria: list[CriterionResult]
    overall_feedback: str

