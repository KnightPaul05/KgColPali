
### `rag_colpali_miniproject.py`

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

# PDF -> images
from pdf2image import convert_from_path

# Vector DB
import faiss

# Optional LLM (Gemini)
import os as _os
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    _HAS_GEMINI = False


def _try_import_colpali():
    try:
        from colpali_engine.models import BiColPali, BiColPaliProcessor
        import torch
        return BiColPali, BiColPaliProcessor, torch
    except Exception:
        return None, None, None


def _try_import_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except Exception:
        return None


class ColPaliEmbedder:
    """
    Embedding multimodal page-level (image + texte) via ColPali.
    Fallback texte-only si ColPali indisponible.
    """
    def __init__(self, device: str = "cuda"):
        BiColPali, BiColPaliProcessor, torch = _try_import_colpali()
        self.fallback = False
        self.device = device

        if BiColPali is not None:
            model_name = "vidore/colpali"
            try:
                self.model = BiColPali.from_pretrained(model_name).to(device)
                self.proc = BiColPaliProcessor.from_pretrained(model_name)
                self.torch = torch
                self.dim = getattr(self.model.config, "projection_dim", 768)
                print(f"[ColPali] OK: {model_name} on {device} (dim={self.dim})")
            except Exception as e:
                print(f"[ColPali] Échec de chargement ({e}). Passage en fallback texte.")
                self._init_fallback()
        else:
            print("[ColPali] Non disponible. Passage en fallback texte.")
            self._init_fallback()

    def _init_fallback(self):
        SentenceTransformer = _try_import_sentence_transformers()
        if SentenceTransformer is None:
            raise RuntimeError("Ni ColPali ni Sentence-Transformers ne sont disponibles. Installez l'un des deux.")
        self.fallback = True
        self.sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384
        print("[Fallback] Sentence-Transformers all-MiniLM-L6-v2 (texte-only)")

    def embed_pages(self, image_paths: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Embeddings pour une liste de chemins d'images (pages PDF).
        Retourne un array (N, dim).
        """
        if self.fallback:
            # Simple placeholder: encode le nom de fichier (à améliorer avec OCR si besoin)
            texts = [Path(p).stem for p in image_paths]
            embs = self.sent_model.encode(texts, normalize_embeddings=True)
            return np.array(embs, dtype=np.float32)

        self.model.eval()
        embs = []
        with self.torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding (ColPali)"):
                batch_paths = image_paths[i:i+batch_size]
                images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.proc(images=images, return_tensors="pt").to(self.device)
                feats = self.model.get_image_features(**inputs)  # (B, D)
                feats = self.torch.nn.functional.normalize(feats, p=2, dim=-1)
                embs.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        if self.fallback:
            embs = self.sent_model.encode(queries, normalize_embeddings=True)
            return np.array(embs, dtype=np.float32)

        self.model.eval()
        with self.torch.no_grad():
            inputs = self.proc(text=queries, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = self.torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype(np.float32)


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200) -> list:
    """
    Convertit un PDF en une liste d'images PNG (une par page).
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    base = Path(pdf_path).stem
    for i, page in enumerate(pages):
        out_path = Path(out_dir) / f"{base}_page_{i+1:04d}.png"
        page.save(out_path, "PNG")
        image_paths.append(str(out_path))
    return image_paths


def build_index(img_dir: str, index_dir: str, device: str = "cuda"):
    """
    Calcule les embeddings de toutes les images de `img_dir` et construit un index FAISS.
    Sauvegarde: index.faiss + mapping.json
    """
    ensure_dirs(index_dir)
    img_paths = sorted([str(p) for p in Path(img_dir).glob("*.png")])
    if not img_paths:
        raise RuntimeError(f"Aucune image trouvée dans {img_dir}.")
    embedder = ColPaliEmbedder(device=device)
    embs = embedder.embed_pages(img_paths)
    dim = embs.shape[1]

    # Cosine similarity ≈ inner product sur vecteurs normalisés
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(Path(index_dir) / "index.faiss"))
    meta = {
        "image_paths": img_paths,
        "dim": int(dim),
        "backend": "colpali" if not embedder.fallback else "fallback-miniLM"
    }
    with open(Path(index_dir) / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Index] Sauvegardé dans {index_dir} ({len(img_paths)} pages).")


def search(index_dir: str, queries: List[str], k: int = 5, device: str = "cuda") -> List[List[Tuple[str, float]]]:
    """
    Recherche top-k pages pour chaque requête.
    Retour: par requête, une liste [(image_path, score), ...]
    """
    index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
    with open(Path(index_dir) / "mapping.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    img_paths = meta["image_paths"]

    embedder = ColPaliEmbedder(device=device)
    q_embs = embedder.embed_queries(queries)
    D, I = index.search(q_embs.astype(np.float32), k)

    results = []
    for qi in range(len(queries)):
        hits = []
        for score, idx in zip(D[qi], I[qi]):
            hits.append((img_paths[int(idx)], float(score)))
        results.append(hits)
    return results


def load_image_snippets(paths_scores: List[Tuple[str, float]]) -> List[str]:
    """
    Prépare des 'snippets' (chemins d'images + score) à injecter dans un prompt LLM.
    """
    snippets = []
    for p, s in paths_scores:
        snippets.append(f"[PAGE_IMG]{p} (score={s:.3f})")
    return snippets


def gemini_answer(query: str, snippets: List[str]) -> str:
    if not _HAS_GEMINI:
        return "(Gemini non installé) Utilisez votre LLM favori avec le prompt ci-dessous.\n" \
               f"Question: {query}\nPages:\n" + "\n".join(snippets)
    api_key = _os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Définissez GOOGLE_API_KEY pour utiliser Gemini."
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    system = (
        "Tu es un assistant RAG. Tu DOIS t'appuyer uniquement sur les pages fournies.\n"
        "Si l'info manque, dis-le. Cite les numéros de page/fichiers quand c'est utile.\n"
        "Les pages sont des images: commente clairement les éléments visuels (tableaux, figures)."
    )
    context = "\n".join(snippets)
    prompt = f"{system}\n\nQuestion: {query}\n\nPages pertinentes:\n{context}\n\nRéponse:"
    resp = model.generate_content(prompt)
    return resp.text


def main():
    parser = argparse.ArgumentParser(description="Mini RAG ColPali (PDF -> FAISS -> QA)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Indexer des PDF en images + embeddings FAISS")
    p_index.add_argument("--pdf-dir", type=str, required=True)
    p_index.add_argument("--img-dir", type=str, required=True)
    p_index.add_argument("--index-dir", type=str, required=True)
    p_index.add_argument("--dpi", type=int, default=200)
    p_index.add_argument("--device", type=str, default="cuda")

    p_query = sub.add_parser("query", help="Rechercher top-k pages pour une question")
    p_query.add_argument("--index-dir", type=str, required=True)
    p_query.add_argument("--img-dir", type=str, required=True)
    p_query.add_argument("--question", type=str, required=True)
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--device", type=str, default="cuda")

    p_answer = sub.add_parser("answer", help="Générer une réponse LLM (ex: Gemini) avec les pages récupérées")
    p_answer.add_argument("--index-dir", type=str, required=True)
    p_answer.add_argument("--img-dir", type=str, required=True)
    p_answer.add_argument("--question", type=str, required=True)
    p_answer.add_argument("--k", type=int, default=5)
    p_answer.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if args.cmd == "index":
        ensure_dirs(args.img_dir, args.index_dir)
        pdfs = sorted([str(p) for p in Path(args.pdf_dir).glob("*.pdf")])
        if not pdfs:
            raise RuntimeError(f"Aucun PDF dans {args.pdf_dir}")
        for pdf in pdfs:
            pdf_to_images(pdf, args.img_dir, dpi=args.dpi)
        build_index(args.img_dir, args.index_dir, device=args.device)

    elif args.cmd == "query":
        results = search(args.index_dir, [args.question], k=args.k, device=args.device)[0]
        print("\n=== Top-k pages ===")
        for rank, (p, s) in enumerate(results, 1):
            print(f"{rank:2d}. {p}   score={s:.3f}")

    elif args.cmd == "answer":
        results = search(args.index_dir, [args.question], k=args.k, device=args.device)[0]
        snippets = load_image_snippets(results)
        ans = gemini_answer(args.question, snippets)
        print("\n=== Réponse LLM ===\n")
        print(ans)


if __name__ == "__main__":
    main()
