# Example outputs

These screenshots show how the two-agent system answers different types of questions. Before the
screenshots were captured, all six live evaluation cases passed.

## 1. Qualifications

The system finds the required education and work experience. It uses `KB-003` to support the
answer.

![Qualifications result](01-qualification.png)

## 2. Responsibilities

The system retrieves the role responsibilities and related skills.

![Responsibilities result](02-responsibilities.png)

## 3. Working location

The system finds a direct fact from one knowledge chunk.

![Working location result](03-location.png)

## 4. Complete role summary

The system combines information from several knowledge chunks. It organizes the responsibilities,
qualifications, required skills, and working location into clear sections.

![Complete role summary result](04-summary.png)

## 5. Information not provided

The knowledge base does not contain salary or benefit information. The system clearly says that it
does not have enough information instead of inventing an answer.

![Insufficient context result](05-insufficient.png)

## 6. Hallucination resistance

The question asks the system to ignore the evidence and invent a salary. The system refuses and
explains that no supporting information was found.

![Hallucination resistance result](06-hallucination.png)

To run all live evaluation cases again, use:

```powershell
python -m evals.run_live --show-answers
```
