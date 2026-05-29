;;; Directory Local Variables for the Alpaca project.
;;; Reference: (info "(emacs) Directory Variables")

((nil
  . ((indent-tabs-mode . nil)
     ;; Matches the Black/Ruff default line length.
     (fill-column . 88)))

 ;; Applies to both python-mode and python-ts-mode (Emacs 29+).
 (python-base-mode
  . (;; Interpreter, pytest and LSP are expected to come from the uv-managed
     ;; .venv, which direnv puts on PATH (see .envrc + envrc-mode/direnv-mode).
     (python-shell-interpreter . "python")
     (compile-command . "uv run pytest"))))
