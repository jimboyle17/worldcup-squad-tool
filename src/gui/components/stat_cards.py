import customtkinter as ctk


class StatCard(ctk.CTkFrame):
    """A single stat card showing a label and value."""

    def __init__(self, master, title: str, value: str, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(corner_radius=10, fg_color=("gray85", "gray20"))

        self.title_label = ctk.CTkLabel(
            self,
            text=title,
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60"),
        )
        self.title_label.pack(padx=14, pady=(12, 2))

        self.value_label = ctk.CTkLabel(
            self,
            text=value,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.value_label.pack(padx=14, pady=(2, 12))

    def update_value(self, value: str):
        self.value_label.configure(text=value)


class StatCardRow(ctk.CTkFrame):
    """A horizontal row of stat cards."""

    def __init__(self, master, stats: list, **kwargs):
        """
        Args:
            stats: List of (title, value) tuples.
        """
        super().__init__(master, **kwargs)
        self.configure(fg_color="transparent")
        self.cards = []

        for i, (title, value) in enumerate(stats):
            card = StatCard(self, title=title, value=str(value))
            card.pack(side="left", padx=(0 if i == 0 else 8, 0), fill="x", expand=True)
            self.cards.append(card)

    def update_stats(self, stats: list):
        for card, (title, value) in zip(self.cards, stats):
            card.title_label.configure(text=title)
            card.update_value(str(value))
