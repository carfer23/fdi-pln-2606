"""Modelos de datos para cartas y estado de recursos."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Carta:
    id: str
    remitente: str
    asunto: str
    cuerpo: str

    @classmethod
    def from_dict(cls, carta_id: str, datos: dict) -> "Carta":
        return cls(
            id=carta_id,
            remitente=datos.get("remi", "Desconocido"),
            asunto=datos.get("asunto", "Sin asunto"),
            cuerpo=datos.get("cuerpo", ""),
        )


@dataclass
class EstadoRecursos:
    faltantes: Dict[str, int] = field(default_factory=dict)
    sobrantes: Dict[str, int] = field(default_factory=dict)

    @property
    def objetivo_cumplido(self) -> bool:
        return len(self.faltantes) == 0

    @property
    def total_faltantes(self) -> int:
        return sum(self.faltantes.values())

    @property
    def total_sobrantes(self) -> int:
        return sum(self.sobrantes.values())
