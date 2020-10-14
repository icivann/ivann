<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <IntegerInc :index=index
                :key="index"
                :value=val
                @increment="increment"
                v-for="(val, index) in value"></IntegerInc>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';

@Component({
  components: { IntegerInc },
})
export default class Vector extends Vue {
  @Prop() value!: [number];

  @Prop({ type: String }) name!: string;

  increment(value: number, index: number) {
    const copy = [...this.value];
    copy[index] = value;

    this.$emit('input', copy);
  }
}
</script>
