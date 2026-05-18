"""Clase base para crear agentes."""


class AgenteBase:
    """Clase base para crear agentes."""

    # --- Obligatorios ---
    def validate_config(self) -> list[str]:
        """Valida la configuración necesaria para el agente. Devuelve una lista de errores encontrados."""
        raise NotImplementedError

    def build_system_prompt(self) -> str:
        """Construye el prompt del sistema."""
        raise NotImplementedError

    def run_loop(self, system_prompt: str) -> None:
        """Ejecuta el bucle principal del agente, recibiendo el prompt del sistema ya construido."""
        raise NotImplementedError

    # --- Opcionales ---
    def on_start(self) -> None:
        """Función opcional que se ejecuta al iniciar el agente, antes de construir el prompt."""
        return None

    def run(self) -> None:
        """Ejecuta el agente, validando la configuración, construyendo el prompt y lanzando el bucle."""
        errors = self.validate_config()
        if errors:
            for error in errors:
                self._print_config_error(error)
            raise RuntimeError(
                "Configuracion incompleta. Revisa las variables de entorno."
            )

        self.on_start()

        system_prompt = self.build_system_prompt()
        self.run_loop(system_prompt)

    def _print_config_error(self, message: str) -> None:
        """Imprime un mensaje de error relacionado con la configuración del agente."""
        print(f"Configuracion inválida: {message}")
