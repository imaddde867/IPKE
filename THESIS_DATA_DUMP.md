# THESIS EXPERIMENTAL DATA SUMMARY
Generated on Tue Nov 25 14:24:12 EET 2025

## Section 5.4: Prompting Strategies (Mistral 7B)
| Strategy | Document | StepF1 | ConstrCov | A-Score | Note |
|---|---|---|---|---|---|

## Section 5.5: Graph Topology Analysis (P3)
*(Data derived from structural analysis of P3 outputs)*

Document,document_id,node_count,step_count,constraint_count,equipment_count,parameter_count,edge_count,density,constraint_density
3M_OEM_SOP,3M_OEM_SOP,105,54,51,0,0,118,0.010805860805860806,0.9444444444444444
DOA_Food_Proc,DOA_Food_Proc,200,102,98,0,0,213,0.005351758793969849,0.9607843137254902
Fire_Safety,op_firesafety_guideline,121,59,62,0,0,122,0.00840220385674931,1.0508474576271187

## Section 5.6: Model Scaling Results
| Model | Strategy | StepF1 | ConstrCov | A-Score |
|---|---|---|---|---|
| Mistral-7B | P3 | 0.619 | 0.750 | 0.699 |
| Llama-8B | P0 | 0.174 | 0.000 | 0.174 |
| Llama-8B | P3 | 0.152 | 0.500 | 0.376 |
| Llama-70B | P0 | 0.170 | 0.000 | 0.187 |
| Llama-70B | P3 | 0.225 | 0.500 | 0.439 |

