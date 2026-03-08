import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';

interface IdeaDetailCtx {
  openIdea: (ideaId: string) => void;
  closeIdea: () => void;
  activeIdeaId: string | null;
}

const Ctx = createContext<IdeaDetailCtx>({
  openIdea: () => {},
  closeIdea: () => {},
  activeIdeaId: null,
});

export function IdeaDetailProvider({ children }: { children: ReactNode }) {
  const [activeIdeaId, setActiveIdeaId] = useState<string | null>(null);
  const openIdea = useCallback((id: string) => setActiveIdeaId(id), []);
  const closeIdea = useCallback(() => setActiveIdeaId(null), []);
  return (
    <Ctx.Provider value={{ openIdea, closeIdea, activeIdeaId }}>
      {children}
    </Ctx.Provider>
  );
}

export function useIdeaDetail() {
  return useContext(Ctx);
}
