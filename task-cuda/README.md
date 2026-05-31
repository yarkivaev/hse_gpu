<markdown><h1>Задание 2. Библиотека для работы с линейными объектами в CUDA</h1>
<p>Ссылка на шаблон: <a href="https://gitlab.akhcheck.ru/task-templates/taskcudatemplate" target="_blank" rel="noopener noreferrer">https://gitlab.akhcheck.ru/task-templates/taskcudatemplate</a></p>
<p>Необходимо продумать библиотеку для работы с векторными и матричными операциями.</p>
<p><strong>Количество баллов указано в относительных величинах! Балл за задание вычисляется по формуле: процент x 1.5</strong></p>
<ol>
<li>(5%) Сложение двух массивов (функция <code>KernelAdd</code>, файл <code>KernelAdd.cuh</code>, реализация в <code>KernelAdd.cu</code>)</li>
<li>(5%) Поэлементное перемножение двух массивов (функция <code>KernelMul</code>, файл <code>KernelMul.cuh</code>, реализация в <code>KernelMul.cu</code>)</li>
<li>(10%) Сложение двух матриц одинакового размера (функция <code>KernelMatrixAdd</code>, файл <code>KernelMatrixAdd.cuh</code>, реализация в <code>KernelMatrixAdd.cu</code>) - матрицы должны быть аллоцированы через механизм двумерных матриц</li>
<li>(10%) Перемножение матрицу на вектор (функция <code>MatrixVectorMul</code>, файл <code>MatrixVectorMul.cuh</code>, реализация в <code>MatrixVectorMul.cu</code>)</li>
<li>(15%) Вычисление скалярного произведения двух векторов (функция <code>ScalarMul</code>, файл <code>ScalarMul.cuh</code>, реализация в <code>KernelScalarMul.cu</code>)</li>
<li>(15% - можно получить после решения задачи 5) Вычисление косинуса угла между двумя векторами (на основе функции скалярного произведения)</li>
<li>(20%) Вычисление произведения двух матриц (функция <code>MatrixMul</code>, файл <code>MatrixMul.cuh</code>, реализация в <code>MatrixMul.cu</code>) через shared memory.</li>
<li>(20%) Реализация функции Filter, которая оставляет только элементы массива, удовлетворяющие соотношению (для этого необходимо реализовать функцию:</li>
</ol>
<pre><code>enum OperationFilterType {
    GT,
    LT
};

__global__ void Filter(float* array, int numElements, OperationFilterType type, float* value)
</code></pre>
<p>Количество элементов в линейных массивах не превосходят <code>2^28</code>, в задачах 5-8 - количество элементов в массивах не превосходит <code>2^20</code>.</p>
<p>Сборка проекта должна осуществляться через CMake.</p>
<p>Реализацию скалярного произведения необходимо произвести двумя способами:</p>
<ul>
<li>реализовать сумму внутри блока (ядро), довести вычисление одним ядром до размера блока, а после этого вычислить с помощью реализованного блока (ScalarMulSumPlusReduction) </li>
<li>реализовать ядро, которое позволит выполнить операцию сложения внутри ядра, вызвав сумму внутри блока два раза (ScalarMulTwoReductions). При этом в CommonKernels.cuh можно добавлять свои ядра (вполне возможно, что внешнее ядро и внутренние ядра могут быть различными)</li>
</ul>
<p>Для каждой операции необходимо будет построить графики зависимости времени вычисления от размера вектора и размера блока. Время работы ядра замеряем посредством библиотеки CUDA (через CUDA Events)</p>
</markdown>