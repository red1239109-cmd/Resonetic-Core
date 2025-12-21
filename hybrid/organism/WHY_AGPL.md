# Why AGPL-3.0?

This document explains the clear reasons and philosophy behind choosing the **GNU AGPL-3.0** license for the **Data Refinery Engine (DRE)**.

In short:
> **DRE is not just a “tool”—it is an engine closer to public infrastructure.**

---

## 1. What DRE Aims to Protect

DRE is not a simple data preprocessing library.  
The project embeds the following as **core values**:

- **Explainability**: Explains why a column was kept or dropped  
- **Auditability**: All important decisions are logged  
- **Governance**: Safety boundaries (thresholds, policies) are unbreakable  
- **Fairness**: Detects structural bias that emerges over time  
- **Reproducibility**: Same input → similar output  

If even one of these is hidden behind a closed service,  
users only see the **result**, while the **basis for judgment and accountability disappears**.

---

## 2. Why Not MIT or Apache-2.0?

### Advantages of MIT / Apache
- Easy to adopt  
- Fast enterprise integration  
- Almost no restrictions  

### Fatal Drawbacks for DRE
- **No obligation to disclose source when running on a server**  
- Possible to remove explainability/fairness logic and still provide the service  
- Enables “closed SaaS with only the shell remaining”

As a result:
> **DRE’s core philosophy (governance, explanation, bias detection) becomes invisible to the outside world.**

---

## 3. What AGPL Protects

AGPL adds just one key clause to GPL:
> “If you provide the program to users over a network, you must also provide the source code.”

Why this matters:
- The moment DRE is **run on a server**,  
- Users gain the **right to see the decision-making logic**  
- Any removal of explainability/audit/fairness becomes immediately visible  

AGPL is not “forced free usage”—  
it is a **safety mechanism that prevents irresponsible use**.

---

## 4. Can Companies Use It?

**Yes—very clearly.**

### Allowed Uses
- Internal analysis pipelines  
- Research and experimentation  
- Internal services  
- Open SaaS (with source disclosure)  

### Not Allowed
- Running DRE on a server while hiding the source code  
- Modified engines with fairness/explainability removed  
- “Sell results only, avoid responsibility” structures  

---

## 5. Is This an Anti-Enterprise License?

No—quite the opposite.  
AGPL protects the following enterprises:

- Teams that **honestly disclose** their services  
- Organizations building **responsible AI/data systems**  
- Enterprises that must comply with **audit and regulatory requirements**

DRE cares more about **“how it is used”** than **“who uses it.”**

---

## 6. Rights and Responsibilities of Contributors

Contributing to this project means agreeing to the following:
- Your contributed code will be distributed under AGPL-3.0  
- If someone uses this code in a service, **users have the right to access that code**  
- Contribution is participation in **public evolution**, not closed monopolization  

---

## 7. Summary (One Sentence)

> **DRE is licensed under AGPL because it is an engine that exposes judgment, not just produces results.**

---

## 8. If You Need a Different License

- Academic/research purposes: Use as-is  
- Commercial/closed-source needs: Separate agreement possible (dual licensing option)  

This project was not created to fight—  
it was created to **go far together with people who share the same direction.**
