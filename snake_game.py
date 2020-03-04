import pygame
import numpy as np
import time

block_size = 32
fps = 10


def main(actions=None, width=10, height=10, bonuses=None):
    width *= block_size
    height *= block_size

    successes, failures = pygame.init()
    print("{} successes, {} failures".format(successes, failures))

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    background = pygame.Surface(screen.get_size())
    background.fill((210, 255, 210))
    background = background.convert()
    screen.blit(background, (0, 0))
    pygame.display.flip()

    snake = [pygame.Rect((width//2, height//2), (block_size, block_size)),
             pygame.Rect((width//2 -block_size, height//2), (block_size, block_size)),
             pygame.Rect((width//2 - 2*block_size, height//2), (block_size, block_size))]
    image = pygame.Surface((block_size, block_size))
    image.fill((20, 50, 20))
    for block in snake:
        screen.blit(image, block)

    bonus = pygame.Rect((block_size * np.random.randint(width//block_size),
                         block_size * np.random.randint(height//block_size)),
                        (block_size, block_size))
    while bonus.collidelist(snake) != -1:
        bonus = pygame.Rect((block_size * np.random.randint(width // block_size),
                             block_size * np.random.randint(height // block_size)),
                            (block_size, block_size))

    bonus_image = pygame.Surface((block_size, block_size))
    bonus_image.fill((255, 0, 0))

    direction = "R"
    score = 0

    if actions is None:
        while True:
            clock.tick(fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and direction != "D":
                        direction = "U"
                    elif event.key == pygame.K_LEFT and direction != "R":
                        direction = "L"
                    elif event.key == pygame.K_RIGHT and direction != "L":
                        direction = "R"
                    elif event.key == pygame.K_DOWN and direction != "U":
                        direction = "D"

            snake, bonus, score, lost = move_in_dir(snake, direction, bonus, score, height, width)
            if lost:
                print("You lost !")
                print("Total score: {}".format(score))
                quit()
            screen.blit(background, (0, 0))
            for block in snake:
                screen.blit(image, block)
            screen.blit(bonus_image, bonus)
            pygame.display.flip()

    else:
        directions = ["L", "U", "R", "D"]
        cd = "R"
        icd = 2
        bn = bonuses[0]
        bn = bn[0] * 32, bn[1] * 32
        old_bon = pygame.Rect(bn, (32, 32))
        for action, bn in zip(actions, bonuses):
            bn = bn[0]*32, bn[1]*32
            bon = pygame.Rect(bn, (32, 32))
            time.sleep(0.2)
            icd += action - 1
            icd = (icd+4) % 4
            cd = directions[icd]
            snake, _, score, lost = move_in_dir(snake, cd, old_bon, score, height, width)
            if lost:
                print("You lost !")
                print("Total score: {}".format(score))
                quit()
            screen.blit(background, (0, 0))
            old_bon = bon
            for block in snake:
                screen.blit(image, block)
            screen.blit(bonus_image, bon)
            pygame.display.flip()



def move_in_dir(snake, direction, bonus, score, height, width):
    chp_x, chp_y = snake[0].x, snake[0].y  # current head position
    if direction == "U":
        snake_head = pygame.Rect((chp_x, chp_y-block_size), (block_size, block_size))
    if direction == "D":
        snake_head = pygame.Rect((chp_x, chp_y + block_size), (block_size, block_size))
    if direction == "L":
        snake_head = pygame.Rect((chp_x - block_size, chp_y), (block_size, block_size))
    if direction == "R":
        snake_head = pygame.Rect((chp_x + block_size, chp_y), (block_size, block_size))

    shp_x, shp_y = snake_head.x, snake_head.y
    if shp_x < 0 or shp_x > width-block_size or shp_y < 0 or shp_y > height-block_size:
        print("OOB")
        lost = True
        return snake, bonus, score, lost

    if bonus.colliderect(snake_head): # The bonus is eaten
        snake_copy = snake[:]
        snake = [snake_head] + snake_copy
        score += 100
        while bonus.collidelist(snake) != -1:
            bonus = pygame.Rect((block_size * np.random.randint(width // block_size),
                                 block_size * np.random.randint(height // block_size)),
                                (block_size, block_size))
    else:
        if snake_head.collidelist(snake[:-1]) != -1:
            print("Collision !")
            lost = True
            return snake, bonus, score, lost
        snake_copy = snake[:-1]
        snake = [snake_head] + snake_copy
    return snake, bonus, score, False


if __name__ == '__main__':
    main()