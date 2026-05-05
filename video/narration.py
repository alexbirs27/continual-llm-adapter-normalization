"""
video/narration.py — Script narativ RO TEHNIC pentru proiectul Continual
Adapter Normalization. 15 scene, ~10 minute.

Tonul: research talk. Fără analogii. Toate numerele sunt din rulările reale
(rezultate `olora_analysis.json`).

REVIZIE (post-poster, mai 2026):
  Narațiunea originală (mai 4) folosea doar cosine similarity (flat vec) și
  concluziona "DA, constrângerea se transferă funcțional". Asta era o citire
  prea optimistă — cosine vectorial e blind la suprapunerea de subspații.
  Posterul a folosit cele 3 metrici (cosine, Gram Frobenius / r, max principal
  angle cosine) și o analiză multi-scale, și a descoperit că:
    - A: ortogonal pe toate metricile ✓
    - B: cosine ≈ 0 dar principal angle = 0.29 (subspațiile coloană se
      suprapun ~30%) ✗
    - ΔW = BA: Frobenius fără rank-normalize este cu un ordin de mărime mai
      mare decât A — ortogonalitatea NU se transferă pe update-ul efectiv
    - BWT colapsează: -0.032 (500 sample-uri/task) → -0.268 (~50% scala
      paper-ului), invers față de ce ai aștepta
  Această narațiune reflectă povestea revizuită.

Structură:
  S1  TITLE       research question — transferă constrângerea pe A la ΔW = BA?
  S2  CL_FORMAL   continual learning formal — secvență de task-uri, ACC, BWT
  S3  CATFORGET   catastrophic forgetting — full fine-tuning
  S4  LORA        recap LoRA: ΔW = B·A
  S5  INCLORA     IncLoRA — adapter izolat per task (upper bound BWT=0)
  S6  OLORA       regularizatorul L_orth = Σ ||A_t A_i^T||_F² (penalty doar pe A)
  S7  GEOM2D      ortogonalizare în 2D — intuiție geometrică
  S8  SUBSPACE    A_t ⊥ span(A_1..A_{t-1}) — claim teoretic, urmează verificare
  S9  SETUP       Qwen2.5-1.5B, rank 8, 28 layers × {q,v}, λ_1 ∈ [0.2, 0.3]
  S10 METHOD      cele 3 metrici × cele 3 quantități (A, B, BA)
  S11 RESULT_A    A — toate metricile ≈ 0, constrângerea respectată
  S12 RESULT_B    B — cosine mic DAR principal angle = 0.29 (revelație)
  S13 RESULT_AB   ΔW = BA — Frobenius fără rank-norm un ordin de mărime mai sus
  S14 SCALE       BWT colapsează de la -0.032 la -0.268 cu mai multe date
  S15 CONCLUSION  nuanțată — A da, BA nu; future work: regularize B
"""

SCRIPT = {
    "s1": "Întrebare de cercetare: regularizatorul OLoRA penalizează produsul matricilor A între task-uri. Dar adapterul efectiv e produsul B ori A. Se transferă ortogonalitatea de pe A pe update-ul efectiv delta W egal cu B ori A? Și e suficient pentru a preveni uitarea catastrofică la scală reală? Validare empirică pe Qwen 2 punct 5, un punct 5 miliarde de parametri, patru task-uri NLP secvențiale.",

    "s2": "Setup formal. O secvență de distribuții de date D unu, D doi, până la D mare T. Modelul vede task-urile o singură dată, în ordine. La pasul t avem acces doar la D t. Țintă: parametri theta t care să performeze rezonabil pe toate task-urile văzute, măsurați prin acuratețea medie ACC și backward transfer BWT — diferența medie între performanța finală pe un task vechi și performanța imediat după antrenare.",

    "s3": "Fine-tuning naiv pe toate parametrii produce uitarea catastrofală. Suprafața de loss pentru task-ul t are minimul deplasat față de cel pentru t minus unu. Optimizatorul migrează acolo, distrugând reprezentările vechi. R i j, acuratețea pe task-ul j după antrenare până la i, scade brutal pe diagonală subdiagonal.",

    "s4": "LoRA. În loc să antrenezi delta W direct, parametrizezi delta W ca produsul B ori A, cu A din R r pe d, B din R d pe r, rang r mult mai mic ca d. Înghețezi W zero, antrenezi doar A și B. Pentru r egal cu opt și d egal cu o mie cinci sute treizeci și șase, reduci parametrii antrenabili de aproape două sute de ori. Important: delta W este produsul, nu A și B separat.",

    "s5": "IncLoRA — baseline. Pentru fiecare task t aloci o pereche nouă A t B t, înghețezi cele vechi. Inferență: cunoști identitatea task-ului, folosești adapterul corespunzător. Zero interferență prin construcție, deci backward transfer zero. Limită superioară de referință — dacă OLoRA depășește IncLoRA, are transfer pozitiv. Dacă nu, OLoRA pierde la uitare ce câștigă în partajare de parametri.",

    "s6": "OLoRA. Loss-ul total devine loss-ul de task plus lambda unu ori suma după i mai mic ca t a normelor Frobenius la pătrat din A t ori A i transpus. Geometric: împinge rândurile lui A t în complementul ortogonal al span-ului rândurilor lui A i, pentru toate i anterioare. Atenție: penalizarea e doar pe A. Matricile B nu sunt regularizate. Asta va deveni important.",

    "s7": "Intuiție în două dimensiuni. Doi vectori A unu și A doi inițial neconstrânși au unghi arbitrar. Adăugarea termenului de penalizare proporțional cu produsul lor scalar la pătrat, derivat în timpul antrenării, îi împinge spre unghi de nouăzeci de grade. La convergență, produsul scalar tinde la zero.",

    "s8": "Generalizare la subspații. Fiecare A t are rang efectiv r. Subspațiul liniar al rândurilor sale e r-dimensional. Constrângerea cere ca acest subspațiu să fie ortogonal pe uniunea celorlalte. Pentru d egal cu o mie cinci sute treizeci și șase, dimensiunea ascunsă în Qwen, și T egal cu patru task-uri cu r egal cu opt — avem treizeci și două de direcții ocupate dintr-o mie cinci sute treizeci și șase. Mai puțin de trei la sută. Dimensional, constrângerea e ușor de satisfăcut. Dar — verifică-mi.",

    "s9": "Setup empiric. Modelul de bază: Qwen 2 punct 5, un punct 5 miliarde, douăzeci și opt de blocuri transformer. Module țintă: doar q proj și v proj din self-attention. Rang LoRA opt, alfa treizeci și doi, lambda unu între zero virgulă doi și zero virgulă trei — mai mic decât valoarea din paper-ul original. Patru task-uri secvențiale: AG News, Amazon Polarity, DBpedia 14, Yahoo Answers Topics. Trei rulări la scale crescătoare: cinci sute de sample-uri pe task, regimul intermediar de două ore, și aproximativ cincizeci la sută din scala paper-ului — opt ore pe GPU.",

    "s10": "Metodologie — auditul geometric. După rularea finală extragem A t și B t per task, per layer, per modul. Pentru fiecare pereche de task-uri i diferit de j calculăm trei metrici. Întâi: cosinusul similarității ca vectori aplatizați — captează direcție medie, e blind la structura de subspațiu. A doua: norma Frobenius a Gram-ului A i A j transpus, normalizată la rang — exact ce minimizează loss-ul OLoRA. A treia: cosinusul unghiului principal maxim între bazele ortonormale ale subspațiilor — captează cea mai apropiată pereche de direcții, indiferent de magnitudine. Trei metrici, trei cantități: A singur, B singur, produsul B A.",

    "s11": "Rezultat A. Heatmap-ul matricilor A pe cele patru task-uri. Toate trei metricile aproape de zero off-diagonal: cosine maxim la zece la minus patru, Gram Frobenius pe rang la zece la minus trei, principal angle la zece la minus doi. Constrângerea OLoRA respectată strict pe A. Atât pe global cât și per layer și per modul — nici un layer pivot. Confirmare: penalizarea funcționează pe ce a fost proiectată.",

    "s12": "Rezultat B. Aici începe povestea reală. Pe cosine flat-vector, B pare ortogonal — maxim cinci virgulă cinci ori zece la minus trei. Concluzie superficială: B e tot mic, fără probleme. Dar cosine flat-vector e blind la subspații. Pe principal angle cosine — metrica care prinde direcția cea mai apropiată dintre subspațiile coloană — valoarea sare la zero virgulă doi nouă. Adică subspațiile B-urilor pentru task-uri diferite împart aproximativ treizeci la sută din direcții. B nu e regularizat și magnitudinea sa crește proporțional cu numărul de pași de antrenare. Doar cosine vectorial nu o vede.",

    "s13": "Rezultatul cheie: produsul delta W egal cu B ori A — direcția update-ului efectiv. Pe layer plot-uri, norma Frobenius fără rank-normalize aplicată pe delta W e cu un ordin de mărime mai mare decât pe A. Cosine flat-vector arată tot zero — pentru că B e mic ca vector — dar Frobenius produsului dezvăluie ce ascunde cosine: când compunem B neregularizat cu A ortogonal, ortogonalitatea nu se conservă. Constrângerea e pe ce factorizezi, nu pe ce contează.",

    "s14": "Consecința la scală. Trei rulări la scale diferite. Short — cinci sute de sample-uri pe task — BWT minus zero virgulă zero trei doi. Aproape zero uitare. Aproape de IncLoRA. Două ore — sample-uri proporționale cu paper-ul, BWT minus zero virgulă zero nouă unu. Uitare moderată. Opt ore — aproximativ cincizeci la sută din paper, BWT minus zero virgulă două șase opt. AG News pierde patruzeci și patru la sută din acuratețe. Mai multe date înseamnă mai multă uitare — invers față de intuiție. Explicația: la scală mai mare, magnitudinea lui B crește, suprapunerea de subspații pe BA crește, interferența crește. Constrângerea pe A e prea slabă pentru a opri asta.",

    "s15": "Concluzie. La granularitatea promisă de paper — matricile A — OLoRA livrează: ortogonalitate sub zece la minus trei pe toate metricile, uniform pe layer și modul. Dar la granularitatea care contează pentru uitare — produsul efectiv B ori A — constrângerea nu se transferă: principal angle pe B la zero virgulă doi nouă, Frobenius pe BA un ordin de mărime peste A. Și empiric, BWT colapsează la scală mai mare. Răspuns la întrebarea de cercetare: nu integral. Constrângerea geometrică pe A se respectă, dar nu se transferă funcțional pe spațiul update-urilor. Direcție viitoare: regularizezi și B, sau constrângi direct norma B ori A în loc de A ori A transpus.",
}
