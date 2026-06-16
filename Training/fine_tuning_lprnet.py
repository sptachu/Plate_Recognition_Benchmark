# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import time
import argparse

# Importy z Twojego projektu
from Models.lprnet.model_lprnet.LPRNet import build_lprnet
from Tools.load_data import LPRDataLoader, CHARS


def fast_decode(preds, chars):
    """Szybkie dekodowanie dla monitoringu (Greedy Decode)"""
    preds = preds.argmax(1)  # Bierzemy najbardziej prawdopodobne znaki
    preds = preds.cpu().numpy()

    decoded_labels = []
    for i in range(preds.shape[0]):
        label = []
        pre_c = -1
        for c in preds[i]:
            if c != pre_c and c != len(chars) - 1:
                label.append(chars[c])
            pre_c = c
        decoded_labels.append("".join(label))
    return decoded_labels

def get_parser():
    parser = argparse.ArgumentParser(description='Trening LPRNet na Europę')
    parser.add_argument('--img_size', default=[94, 24], help='Wymiary wejściowe')
    parser.add_argument('--train_dir', default='./data/cropped', help='Folder z wycinkami do treningu')
    parser.add_argument('--test_dir', default='./data/UC3M-LP/test', help='Folder z wycinkami do testów')
    parser.add_argument('--epochs', default=100, type=int, help='Liczba epok')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--lpr_max_len', default=8, type=int, help='Max długość tablicy')
    parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='Model bazowy')
    parser.add_argument('--save_dir', default='./weights/europe', help='Gdzie zapisywać nowe wagi')
    parser.add_argument('--cuda', default=False, type=bool, help='Użyj GPU jeśli dostępne')
    return parser.parse_args()


def collate_fn(batch):
    """Funkcja do łączenia obrazków i etykiet o różnych długościach"""
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    return (torch.stack(imgs, 0), torch.from_numpy(np.array(labels).astype(np.float32)), lengths)


def train():
    args = get_parser()
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    device = torch.device("cpu")
    print(f"[*] Trening na urządzeniu: {device}")

    # 1. Budowa modelu
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=True, class_num=len(CHARS), dropout_rate=0.5)
    lprnet.to(device)

    # 2. Transfer Learning: Ładowanie wag z filtrowaniem
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"[*] Ładowanie wag przedtreningowych z {args.pretrained_model}...")
        pretrained_dict = torch.load(args.pretrained_model, map_location=device)
        model_dict = lprnet.state_dict()
        # Filtrujemy tylko te warstwy, które mają ten sam rozmiar (Backbone)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        lprnet.load_state_dict(model_dict)
        print(f"[+] Załadowano {len(pretrained_dict)} warstw (Transfer Learning OK).")

    # 3. DataLoaders
    train_dataset = LPRDataLoader([args.train_dir], args.img_size, args.lpr_max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn)

    # 4. Optimizer i Loss (CTC Loss)
    optimizer = optim.Adam(lprnet.parameters(), lr=args.lr)
    # CTCLoss jest kluczowa dla OCR bez segmentacji znaków
    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

    # --- PĘTLA TRENINGOWA ---
    lprnet.train()
    for epoch in range(args.epochs):
        loss_val = 0
        t_start = time.time()

        for i, (imgs, labels, lengths) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = lprnet(imgs)  # Kształt: [N, C, T] -> [Batch, Classes, Time_Steps]

            # Przygotowanie pod CTCLoss: wejście musi mieć kształt [T, N, C]
            logits = logits.permute(2, 0, 1)
            logits = logits.log_softmax(2)

            # Obliczanie długości wejściowych i wyjściowych
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=logits.size(0), dtype=torch.long)
            target_lengths = torch.from_numpy(np.array(lengths)).long()

            # Obliczanie straty
            loss = ctc_loss(logits, labels, input_lengths, target_lengths)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"[!] Warning: Loss is {loss}, skipping batch")
                continue

            loss.backward()
            optimizer.step()
            # --- WIZUALNY MONITORING CO 100 BATCHY ---
            if i % 1000 == 0:
                lprnet.eval()  # Tryb ewaluacji (wyłącza Dropout)
                with torch.no_grad():
                    test_logits = lprnet(imgs[:5].to(device))  # Testujemy pierwsze 5 obrazków z batcha
                    predictions = fast_decode(test_logits, CHARS)

                    # Odczytujemy prawdziwe etykiety (ground truth)
                    targets = []
                    curr_pos = 0
                    for length in lengths[:5]:
                        t_label = "".join([CHARS[int(x)] for x in labels[curr_pos:curr_pos + length]])
                        targets.append(t_label)
                        curr_pos += length

                    print("\n" + "=" * 50)
                    print(f"BATCH {i} | MONITORING POSTĘPÓW:")
                    print(f"{'TRUE LABEL':<15} | {'PREDICTION':<15} | STATUS")
                    print("-" * 50)
                    for gt, pred in zip(targets, predictions):
                        status = "✅" if gt == pred else "❌"
                        print(f"{gt:<15} | {pred:<15} | {status}")
                    print("=" * 50 + "\n")

                lprnet.train()  # Wracamy do trybu treningu

            loss_val += loss.item()

        avg_loss = loss_val / len(train_loader)
        t_end = time.time()

        print(f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.4f} | Time: {t_end - t_start:.2f}s")

        # Zapisywanie modelu co 10 epok
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.save_dir, f"LPRNet_Euro_Epoch_{epoch + 1}.pth")
            torch.save(lprnet.state_dict(), save_path)
            print(f"[*] Zapisano model: {save_path}")

    print("[+] Trening zakończony!")


if __name__ == "__main__":
    train()