$py = ".venv\Scripts\python.exe"

# Case 01: full girl → walk girl
& $py run_swap.py --target "images (1).jpg" --donor "images (3).jpg" --output "assets/01_fullgirl_to_walkgirl"

# Case 02: gym man → blue shirt
& $py run_swap.py --target "cross_arms_man.jpg" --donor "sample_cross_arms.jpg" --output "assets/02_gymman_to_blueshirt"

# Case 03: glasses man → sweater man
& $py run_swap.py --target "images.jpg" --donor "images (2).jpg" --output "assets/03_glassesman_to_sweaterman"

# Case 04: suit man → girl dress
& $py run_swap.py --target "stand_suit.jpg" --donor "girl_dress.jpg" --output "assets/04_suitman_to_girldress"

# Case 05: short man (multi-person) → white tee/hoodie
& $py run_swap.py --target "888piC4i888piCfHJ.jpg" --target_person 1 --donor "sample_casual_stand.jpg" --output "assets/05_shortman_to_whitetee"

# Case 06: hoodie girl → skirt girl
& $py run_swap.py --target "sample_casual_stand.jpg" --donor "girl_dress.jpg" --output "assets/06_hoodiegirl_to_skirtgirl"

# Case 07: front man → side man
& $py run_swap.py --target "sample_full_man.jpg" --donor "sample_suit_pose.jpg" --output "assets/07_frontman_to_sideman"

# Case 08: sofa man → stand sweater
& $py run_swap.py --target "126488.jpg" --donor "images (2).jpg" --output "assets/08_sofa_man_to_stand_sweater"

# Case 09: sofa girl → walk girl
& $py run_swap.py --target "126492.jpg" --donor "images (3).jpg" --output "assets/09_sofa_girl_to_walk_girl"

# Case 10: stand girl → sofa girl
& $py run_swap.py --target "images (3).jpg" --donor "126492.jpg" --output "assets/10_stand_girl_to_sofa_girl"

# Case 11: chair girl → walk skirt
& $py run_swap.py --target "6c91e12f686844d0b3a081695dc41ea7_th.jpg" --donor "images (3).jpg" --output "assets/11_chair_girl_to_walk_skirt"

# Case 12: walk skirt → chair girl
& $py run_swap.py --target "images (3).jpg" --donor "6c91e12f686844d0b3a081695dc41ea7_th.jpg" --output "assets/12_walk_skirt_to_chair_girl"

# Case 13: back girl → back man
& $py run_swap.py --target "images (4).jpg" --donor "images (5).jpg" --output "assets/13_backgirl_to_backman"

# Case 14: back man → back girl
& $py run_swap.py --target "images (5).jpg" --donor "images (4).jpg" --output "assets/14_backman_to_backgirl"

# Case 15: korean man → dual guy (multi-person target)
& $py run_swap.py --target "雙人照-920x680-1.jpg" --target_person 1 --donor "sample_suit_pose.jpg" --output "assets/15_koreanman_to_dual_guy"

# Case 16: walk girl → dual girl (multi-person target)
& $py run_swap.py --target "雙人照-920x680-1.jpg" --target_person 2 --donor "images (3).jpg" --output "assets/16_walkgirl_to_dual_girl"

Write-Host "`n>> ALL 16 CASES DONE!"
