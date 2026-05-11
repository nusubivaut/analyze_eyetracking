import pygame
import sys
import joblib
import numpy as np
import pandas as pd
from pygame.locals import *
import pygame_gui
from pygame_gui.elements import UITextEntryLine, UIButton, UILabel
import os

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================

model_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(model_path)

print("Загрузка модели...")
model = joblib.load(os.path.join(model_path, 'fixation_model.pkl'))
metadata = joblib.load(os.path.join(model_path, 'model_metadata.pkl'))
features = metadata['features']
print(f"Модель загружена. Признаки: {features}")

# ============================================
# КОНСТАНТЫ ЭКРАНА
# ============================================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
YELLOW = (225, 225, 120)
ORANGE = (205, 150, 50)
RED = (205, 50, 50)
GREEN = (100, 205, 100)
BLUE = (50, 50, 205)

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================
def get_word_features(word, age, iq, reading_speed, position_ratio=0.5):
    length = len(word)
    freq = 50.0 * length
    log_freq = np.log(freq + 0.001)
    predictability = max(0.1, 1.0 - length / 20.0)
    pos_encoded = 0
    input_data = pd.DataFrame([[
        age, iq, reading_speed, length, log_freq, 
        predictability, position_ratio, pos_encoded
    ]], columns=features)
    return input_data

def predict_word_difficulty(word, age, iq, reading_speed, position_ratio=0.5):
    X = get_word_features(word, age, iq, reading_speed, position_ratio)
    pred_log = model.predict(X)[0]
    pred_time = np.expm1(pred_log)
    if pred_time > 700:
        color = RED
    elif pred_time > 500:
        color = ORANGE
    elif pred_time > 350:
        color = YELLOW
    else:
        color = GREEN
    return color, pred_time

# ============================================
# ЭКРАН ВВОДА ПАРАМЕТРОВ (работает)
# ============================================
def input_screen_gui(screen, font):
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    age_input = UITextEntryLine(relative_rect=pygame.Rect(450, 150, 200, 40), manager=manager)
    speed_input = UITextEntryLine(relative_rect=pygame.Rect(450, 230, 200, 40), manager=manager)
    
    age_input.set_text('10')
    speed_input.set_text('150')
    
    next_button = UIButton(relative_rect=pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT - 100, 200, 50),
                           text='ДАЛЕЕ', manager=manager)
    
    error_label = None
    clock = pygame.time.Clock()
    
    while True:
        time_delta = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == next_button:
                try:
                    age = int(age_input.get_text())
                    iq = 28
                    speed = int(speed_input.get_text())
                    if 7 <= age <= 18 and 20 <= iq <= 36 and 50 <= speed <= 300:
                        manager.clear_and_reset()
                        return age, iq, speed
                    else:
                        if error_label:
                            error_label.kill()
                        error_label = UILabel(relative_rect=pygame.Rect(150, 400, 400, 30),
                                              text="Ошибка: значения вне допустимого диапазона!",
                                              manager=manager, object_id='#error')
                except ValueError:
                    if error_label:
                        error_label.kill()
                    error_label = UILabel(relative_rect=pygame.Rect(150, 400, 300, 30),
                                          text="Ошибка: введите целые числа!",
                                          manager=manager, object_id='#error')
            manager.process_events(event)
        
        manager.update(time_delta)
        screen.fill(WHITE)
        screen.blit(font.render("НАСТРОЙКА ПРОФИЛЯ ЧИТАТЕЛЯ", True, BLUE), (SCREEN_WIDTH//2 - 150, 30))
        screen.blit(pygame.font.Font(None, 24).render("Возраст (7-18):", True, BLACK), (150, 160))
        screen.blit(pygame.font.Font(None, 24).render("Скорость чтения (50-300):", True, BLACK), (150, 240))
        manager.draw_ui(screen)
        pygame.display.flip()

# ============================================
# ЭКРАН ВВОДА ТЕКСТА (многострочный)
# ============================================
def text_input_screen_gui(screen, font, age, iq, reading_speed):
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    text_entry = UITextEntryLine(relative_rect=pygame.Rect(50, 100, SCREEN_WIDTH-100, 300),
                                          manager=manager)
    text_entry.set_text("Это пример текста. Напишите или вставьте свой текст.\nПрограмма подсветит сложные слова.")
    
    analyze_button = UIButton(relative_rect=pygame.Rect(SCREEN_WIDTH//2 - 160, SCREEN_HEIGHT - 80, 150, 40),
                              text='АНАЛИЗИРОВАТЬ', manager=manager)
    clear_button = UIButton(relative_rect=pygame.Rect(SCREEN_WIDTH//2 + 10, SCREEN_HEIGHT - 80, 150, 40),
                            text='ОЧИСТИТЬ', manager=manager)
    
    clock = pygame.time.Clock()
    while True:
        time_delta = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == analyze_button:
                    full_text = text_entry.get_text()
                    manager.clear_and_reset()
                    return full_text
                if event.ui_element == clear_button:
                    text_entry.set_text("")
            manager.process_events(event)
        
        manager.update(time_delta)
        screen.fill(WHITE)
        screen.blit(font.render("ВВЕДИТЕ ТЕКСТ ДЛЯ АНАЛИЗА", True, BLUE), (SCREEN_WIDTH//2 - 150, 20))
        info = pygame.font.Font(None, 24).render(f"Возраст: {age} | Скорость: {reading_speed}", True, DARK_GRAY)
        screen.blit(info, (SCREEN_WIDTH//2 - info.get_width()//2, 60))
        manager.draw_ui(screen)
        pygame.display.flip()

# ============================================
# ЭКРАН РЕЗУЛЬТАТА
# ============================================
def result_screen(screen, default_font, font_text, text, age, iq, reading_speed):
    clock = pygame.time.Clock()
    sentences = text.split('.')
    word_data = []
    for i, sentence in enumerate(sentences):
        sentence = sentence + '.'
        words = sentence.replace('\n', ' ').split(' ')
        for i, word in enumerate(words):
            if word == '':
                continue
            clean_word = word.strip('.,!?;:()"\'')
            if clean_word:
                color, pred_time = predict_word_difficulty(clean_word, age, iq, reading_speed, i/(len(words)+1))
            else:
                color, pred_time = BLACK, 0
            word_data.append((word, color, pred_time))
    
    margin = 50
    line_height = font_text.get_height() + 10
    x = margin
    y = 150
    max_width = SCREEN_WIDTH - 2 * margin
    text_bg_rect = pygame.Rect(margin-10, 130, SCREEN_WIDTH-2*margin+20, SCREEN_HEIGHT-200)
    scroll_y = 0
    max_scroll = 0
    back_rect = pygame.Rect(50, SCREEN_HEIGHT - 50, 120, 40)
    exit_rect = pygame.Rect(SCREEN_WIDTH - 170, SCREEN_HEIGHT - 50, 120, 40)
    
    while True:
        screen.fill(WHITE)
        screen.blit(font_text.render("РЕЗУЛЬТАТ АНАЛИЗА ТЕКСТА", True, BLUE), (SCREEN_WIDTH//2 - 150, 20))
        info_font = pygame.font.Font(None, 24)
        info = info_font.render(f"Возраст: {age} | Скорость чтения: {reading_speed} сл/мин", True, DARK_GRAY)
        screen.blit(info, (SCREEN_WIDTH//2 - info.get_width()//2, 60))
        
        legend_y = 10
        for col, lbl in [(GREEN, "лёгкое (<350 мс)"), (YELLOW, "среднее (350-500)"),
                         (ORANGE, "сложное (500-700)"), (RED, "очень сложное (>700)")]:
            pygame.draw.rect(screen, col, (margin, legend_y, 20, 20))
            screen.blit(info_font.render(lbl, True, BLACK), (margin+25, legend_y))
            legend_y += 25
        
        pygame.draw.rect(screen, GRAY, text_bg_rect, 2)
        clip_rect = text_bg_rect.inflate(-4, -4)
        old_clip = screen.get_clip()
        screen.set_clip(clip_rect)
        
        current_x = x
        current_y = y + scroll_y
        for word, color, _ in word_data:
            word_surf = font_text.render(word, True, color)
            w = word_surf.get_width()
            if current_x + w > x + max_width:
                current_x = x
                current_y += line_height
            if current_y + line_height > text_bg_rect.y and current_y < text_bg_rect.y + text_bg_rect.height:
                screen.blit(word_surf, (current_x, current_y))
            current_x += w + font_text.size(' ')[0]
        max_scroll = max(0, (current_y + line_height) - (text_bg_rect.y + text_bg_rect.height))
        screen.set_clip(old_clip)
        
        pygame.draw.rect(screen, BLUE, back_rect)
        pygame.draw.rect(screen, DARK_GRAY, exit_rect)
        screen.blit(default_font.render("НАЗАД", True, WHITE), (back_rect.x+23, back_rect.y+10))
        screen.blit(default_font.render("ВЫХОД", True, WHITE), (exit_rect.x+20, exit_rect.y+10))
        
        if max_scroll > 0:
            scroll_ratio = -scroll_y / max_scroll
            bar_rect = pygame.Rect(SCREEN_WIDTH-20, text_bg_rect.y+10, 10, text_bg_rect.height-20)
            pygame.draw.rect(screen, LIGHT_BLUE, bar_rect)
            handle_h = max(30, bar_rect.height * 0.1)
            handle_y = bar_rect.y + scroll_ratio * (bar_rect.height - handle_h)
            pygame.draw.rect(screen, BLUE, (bar_rect.x, handle_y, bar_rect.width, handle_h))
        
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                if back_rect.collidepoint(event.pos):
                    return
                if exit_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
            if event.type == MOUSEWHEEL:
                scroll_y -= event.y * 20
                scroll_y = max(-max_scroll, min(0, scroll_y))
        clock.tick(30)

# ============================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Ассистент чтения - адаптация экранных форм")
    font = pygame.font.Font(None, 28)
    font_result = pygame.font.Font(None, 60)
    
    while True:
        age, iq, reading_speed = input_screen_gui(screen, font)
        text = text_input_screen_gui(screen, font, age, iq, reading_speed)
        result_screen(screen, font, font_result, text, age, iq, reading_speed)

if __name__ == "__main__":
    main()