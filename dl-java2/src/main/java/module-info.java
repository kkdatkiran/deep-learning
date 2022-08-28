module in.dljava {
	exports in.dljava;

	requires jdk.incubator.vector;
	requires static lombok;
	requires nd4j.api;
	requires deeplearning4j.nn;
	requires nd4j.common;
	requires slf4j.api;
}