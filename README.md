# METODOLOGIA DE GERAÇÃO DE DADOS APLICADA À IMAGENS DE BIÓPSIA DE CÂNCER DE MAMA

Desenvolvemos uma metodologia de geração de dados aplicada a imagens de biópsia de câncer de mama utilizando o dataset **BreakHis 200x**, focando na classe **benigna** para treinar um modelo de geração baseado em **guided diffusion**.

## Etapas do Processo

1. **Treinamento do Modelo de Geração:**  
   Utilizamos o modelo guided diffusion para gerar imagens sintéticas a partir das imagens reais do dataset BreakHis 200x.

2. **Filtragem de Imagens Repetidas:**  
   Aplicamos um método de exclusão para remover imagens duplicadas e garantir diversidade nos dados sintéticos.

3. **Cálculo do FID (Frechet Inception Distance):**  
   Calculamos o FID para avaliar a qualidade e a similaridade das imagens geradas em relação às imagens reais do dataset original.

4. **Treinamento do Modelo de Classificação:**  
   Utilizamos as imagens sintéticas geradas para treinar um modelo de classificação de imagens, melhorando a robustez e a precisão do modelo.

Essa abordagem visa enriquecer o dataset inicial, melhorar a performance do modelo de classificação e explorar novas possibilidades na geração de imagens médicas. Caso deseje mais detalhes sobre o processo, consulte a documentação técnica ou entre em contato conosco.

