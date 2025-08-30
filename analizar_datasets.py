import os
import pandas as pd

def analyze_dataset(images_dir, labels_base_dir, dataset_name):
    splits = ['train', 'valid', 'test']
    rows = []

    totals = {'imgs': 0, 'segs': 0, 'bg': 0}

    for split in splits:
        img_dir = os.path.join(images_dir, split)
        lbl_dir = os.path.join(labels_base_dir, split)

        if not os.path.isdir(img_dir):
            print(f"[ADVERTENCIA] No existe el directorio de imágenes: {img_dir}")
            continue

        # contar imágenes
        imgs = [f for f in os.listdir(img_dir)
                if os.path.isfile(os.path.join(img_dir, f))]
        n_imgs = len(imgs)

        # contar archivos .txt
        if os.path.isdir(lbl_dir):
            lbls = [f for f in os.listdir(lbl_dir)
                    if f.lower().endswith('.txt')]
        else:
            lbls = []
        n_lbl_files = len(lbls)

        # contar líneas (segmentaciones)
        n_segs = 0
        for txt in lbls:
            txt_path = os.path.join(lbl_dir, txt)
            try:
                with open(txt_path, 'r') as fh:
                    n_segs += sum(1 for _ in fh)
            except Exception as e:
                print(f"[ERROR] No se pudo leer {txt_path}: {e}")

        # imágenes de fondo
        n_bg = n_imgs - n_lbl_files

        rows.append({
            'Dataset': dataset_name,
            'Tarea': split.capitalize(),
            'Num. imgs.': n_imgs,
            'Num. segm.': n_segs,
            'Img. fondo': n_bg
        })

        totals['imgs'] += n_imgs
        totals['segs'] += n_segs
        totals['bg'] += n_bg

    # Totales
    rows.append({
        'Dataset': dataset_name,
        'Tarea': 'Total',
        'Num. imgs.': totals['imgs'],
        'Num. segm.': totals['segs'],
        'Img. fondo': totals['bg']
    })

    return pd.DataFrame(rows, columns=['Dataset', 'Tarea', 'Num. imgs.', 'Num. segm.', 'Img. fondo'])

if __name__ == '__main__':
    images_dir = "SalmonesV5_Complete/images"
    datasets = {
        "SalmonesV5": "SalmonesV5_Complete/labels",    
    }

    all_reports = []

    for name, path in datasets.items():
        print(f"Procesando {name}...")
        report_df = analyze_dataset(images_dir, path, name)
        all_reports.append(report_df)

    # Combinar todo en un único DataFrame
    combined_df = pd.concat(all_reports, ignore_index=True)

    # Imprimir como tabla LaTeX
    print("\n========= TABLA LATEX =========\n")
    latex_table = combined_df.to_latex(index=False, caption="Composición de los datasets", label="tab:dataset_composition")
    print(latex_table)
