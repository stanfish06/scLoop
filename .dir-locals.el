;; For emacs local setting
;; change venv bin path if needed
;; pyright needs to find python interpreter to check imports
((nil
  .
  ((eval .
         (let ((venv "/home/stanfish/Git/scloop/.venv/bin"))
           (setenv "PATH" (concat venv ":" (getenv "PATH")))
           (setq exec-path (cons venv exec-path))
           (setq python-shell-interpreter (concat venv "/python"))))))
 (python-mode
  .
  ((eval .
         (let ((venv "/home/stanfish/Git/scloop/.venv/bin"))
           (setenv "PATH" (concat venv ":" (getenv "PATH")))
           (setq exec-path (cons venv exec-path))
           (setq python-shell-interpreter (concat venv "/python")))))))
